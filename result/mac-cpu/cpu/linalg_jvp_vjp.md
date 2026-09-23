# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`
- Benchmark commit: `8595ce266fa34a87659e72c9b14195383343df95`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_103427/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_103427`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_103427`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_103427/cpu_ops_t1_20260923_103427.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_103427/linalg_jvp_vjp_t1_20260923_103427.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 72.311 ± 0.242 | 80.435 ± 0.745 | 134.421 ± 0.594 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.970 ± 0.047 | 3.066 ± 0.056 | 3.914 ± 0.035 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.506 ± 0.055 | 14.412 ± 0.144 | 21.507 ± 0.117 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 71.564 ± 1.500 | 69.770 ± 0.370 | 133.219 ± 0.501 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.886 ± 0.018 | 2.817 ± 0.030 | 3.922 ± 0.041 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.141 ± 0.045 | 12.833 ± 0.081 | 21.273 ± 0.176 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 27.166 ± 0.820 | 58.127 ± 2.186 | 86.233 ± 0.400 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.659 ± 0.006 | 1.304 ± 0.018 | 1.743 ± 0.028 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 3.971 ± 0.031 | 8.383 ± 0.052 | 11.892 ± 0.091 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 28.206 ± 0.206 | 31.167 ± 0.682 | 87.195 ± 0.968 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.683 ± 0.006 | 0.768 ± 0.014 | 1.722 ± 0.006 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.004 ± 0.026 | 4.645 ± 0.070 | 11.839 ± 0.101 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 49.295 ± 0.455 | 56.899 ± 0.535 | 134.856 ± 1.010 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.780 ± 0.013 | 1.918 ± 0.015 | 3.165 ± 0.011 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.562 ± 0.057 | 9.848 ± 0.071 | 18.879 ± 0.074 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 49.811 ± 0.713 | 54.337 ± 1.685 | 137.258 ± 0.596 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.834 ± 0.008 | 1.840 ± 0.021 | 3.147 ± 0.020 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.726 ± 0.019 | 9.321 ± 0.116 | 19.587 ± 0.031 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 5.213 ± 0.043 | 5.645 ± 0.096 | 5.153 ± 0.073 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.253 ± 0.003 | 0.294 ± 0.011 | 0.216 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.100 ± 0.019 | 1.184 ± 0.063 | 1.040 ± 0.034 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 5.050 ± 0.022 | 9.379 ± 0.086 | 4.789 ± 0.085 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.247 ± 0.002 | 0.385 ± 0.011 | 0.217 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.058 ± 0.025 | 1.881 ± 0.042 | 0.985 ± 0.017 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 117.217 ± 0.418 | 133.723 ± 1.561 | 177.539 ± 0.348 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.046 ± 0.039 | 5.507 ± 0.049 | 6.085 ± 0.030 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 23.009 ± 0.088 | 25.876 ± 0.086 | 30.993 ± 0.205 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 116.310 ± 0.513 | 112.534 ± 0.699 | 176.879 ± 0.315 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 4.986 ± 0.027 | 4.976 ± 0.028 | 6.031 ± 0.039 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 22.916 ± 0.055 | 22.538 ± 0.084 | 30.751 ± 0.081 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.039 ± 0.001 | 0.050 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.065 ± 0.002 | 0.074 ± 0.001 | 0.044 ± 0.002 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.031 ± 0.001 | 0.043 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.044 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.063 ± 0.000 | 0.068 ± 0.001 | 0.044 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.023 ± 0.001 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.033 ± 0.000 | 0.060 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.039 ± 0.000 | 0.074 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.001 | 0.058 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.033 ± 0.000 | 0.048 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.038 ± 0.000 | 0.056 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.024 ± 0.001 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.031 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.050 ± 0.001 | 0.048 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.064 ± 0.001 | 0.055 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.039 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.041 ± 0.000 | 0.045 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.051 ± 0.002 | 0.047 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.039 ± 0.002 | 0.037 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.066 ± 0.000 | 0.054 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.040 ± 0.000 | 0.038 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.000 | 0.044 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.024 ± 0.001 | 0.082 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.023 ± 0.000 | 0.081 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.032 ± 0.002 | 0.089 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.023 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.023 ± 0.001 | 0.082 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.023 ± 0.000 | 0.038 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.022 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.032 ± 0.001 | 0.049 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.022 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.040 ± 0.000 | 0.074 ± 0.001 | 0.027 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.080 ± 0.000 | 0.123 ± 0.002 | 0.075 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.015 ± 0.000 | 0.048 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.027 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.031 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.080 ± 0.001 | 0.103 ± 0.002 | 0.076 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_105726/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_105726`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_105726`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_105726/cpu_ops_t4_20260923_105726.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_105726/linalg_jvp_vjp_t4_20260923_105726.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 69.264 ± 0.141 | 79.046 ± 0.275 | 84.599 ± 0.809 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 2.921 ± 0.029 | 3.044 ± 0.048 | 3.052 ± 0.027 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 12.990 ± 0.066 | 14.081 ± 0.090 | 15.116 ± 0.065 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 67.779 ± 0.193 | 68.877 ± 0.223 | 85.003 ± 0.568 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 2.827 ± 0.022 | 2.798 ± 0.021 | 3.018 ± 0.028 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 12.643 ± 0.045 | 12.689 ± 0.048 | 15.089 ± 0.093 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 27.038 ± 0.164 | 53.674 ± 0.577 | 35.011 ± 0.388 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.739 ± 0.005 | 1.351 ± 0.016 | 0.753 ± 0.006 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.038 ± 0.032 | 7.557 ± 0.030 | 5.384 ± 0.127 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 27.529 ± 0.089 | 29.623 ± 0.156 | 35.767 ± 0.387 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.728 ± 0.007 | 0.848 ± 0.009 | 0.817 ± 0.012 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 3.920 ± 0.021 | 4.202 ± 0.027 | 5.453 ± 0.145 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 46.479 ± 2.754 | 51.061 ± 0.249 | 59.802 ± 0.532 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.858 ± 0.009 | 1.843 ± 0.022 | 1.788 ± 0.037 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.015 ± 0.060 | 8.386 ± 0.058 | 9.571 ± 0.672 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 45.985 ± 0.190 | 48.811 ± 0.545 | 63.683 ± 0.317 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.849 ± 0.009 | 1.802 ± 0.029 | 1.877 ± 0.012 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.071 ± 0.044 | 8.117 ± 0.073 | 10.289 ± 0.128 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.015 ± 0.088 | 5.215 ± 0.132 | 5.047 ± 0.167 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.282 ± 0.004 | 0.293 ± 0.003 | 0.220 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.077 ± 0.027 | 1.038 ± 0.030 | 1.100 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 4.879 ± 0.058 | 8.841 ± 0.113 | 4.746 ± 0.082 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.272 ± 0.005 | 0.406 ± 0.005 | 0.216 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.045 ± 0.008 | 1.717 ± 0.014 | 1.096 ± 0.045 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 116.195 ± 0.467 | 131.191 ± 0.857 | 127.753 ± 2.143 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.012 ± 0.030 | 5.488 ± 0.063 | 5.289 ± 0.063 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 22.902 ± 0.067 | 25.116 ± 0.496 | 24.413 ± 0.132 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 115.235 ± 0.458 | 111.634 ± 0.538 | 125.339 ± 0.828 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.011 ± 0.059 | 4.922 ± 0.039 | 5.224 ± 0.032 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 22.842 ± 0.124 | 22.431 ± 0.179 | 24.207 ± 0.071 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.039 ± 0.001 | 0.050 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.067 ± 0.002 | 0.074 ± 0.001 | 0.044 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.031 ± 0.001 | 0.042 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.039 ± 0.001 | 0.045 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.028 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.064 ± 0.000 | 0.069 ± 0.002 | 0.044 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.036 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.033 ± 0.000 | 0.140 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.024 ± 0.001 | 0.129 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.042 ± 0.000 | 0.150 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.000 | 0.131 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.133 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.033 ± 0.001 | 0.100 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.024 ± 0.001 | 0.095 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.039 ± 0.001 | 0.109 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.095 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.031 ± 0.000 | 0.099 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.051 ± 0.002 | 0.076 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.040 ± 0.001 | 0.069 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.065 ± 0.005 | 0.084 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.040 ± 0.001 | 0.070 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.041 ± 0.000 | 0.075 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.052 ± 0.001 | 0.082 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.040 ± 0.001 | 0.076 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.066 ± 0.001 | 0.088 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.040 ± 0.000 | 0.076 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.048 ± 0.000 | 0.078 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.024 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.024 ± 0.001 | 0.080 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.034 ± 0.000 | 0.089 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.023 ± 0.001 | 0.079 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.024 ± 0.000 | 0.083 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.023 ± 0.001 | 0.038 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.023 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.032 ± 0.001 | 0.047 ± 0.005 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.023 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.023 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.040 ± 0.000 | 0.074 ± 0.001 | 0.028 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.001 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.081 ± 0.000 | 0.123 ± 0.002 | 0.076 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.015 ± 0.000 | 0.048 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.039 ± 0.001 | 0.056 ± 0.000 | 0.029 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.080 ± 0.002 | 0.102 ± 0.001 | 0.077 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

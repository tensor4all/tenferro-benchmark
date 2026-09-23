# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`
- Benchmark commit: `9f63fb7704feec767fda97251865741714f3926f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_173045/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_173045`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_173045`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_173045/cpu_ops_t1_20260923_173045.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_173045/linalg_jvp_vjp_t1_20260923_173045.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 74.312 ± 5.759 | 81.558 ± 1.560 | 135.062 ± 0.430 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.935 ± 0.037 | 3.048 ± 0.032 | 3.971 ± 0.204 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.324 ± 0.033 | 14.686 ± 0.289 | 22.272 ± 2.170 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 71.536 ± 2.843 | 70.180 ± 0.565 | 133.775 ± 0.362 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.912 ± 0.042 | 2.809 ± 0.018 | 3.933 ± 0.034 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.203 ± 0.069 | 13.336 ± 0.957 | 21.693 ± 0.093 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 28.572 ± 0.915 | 60.216 ± 2.437 | 86.434 ± 0.223 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.661 ± 0.007 | 1.303 ± 0.020 | 1.732 ± 0.023 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 3.969 ± 0.070 | 9.260 ± 1.334 | 12.048 ± 0.243 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 28.163 ± 0.419 | 37.567 ± 3.016 | 87.997 ± 0.545 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.683 ± 0.003 | 0.768 ± 0.006 | 1.740 ± 0.021 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 3.992 ± 0.038 | 4.701 ± 0.024 | 11.948 ± 0.207 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 48.876 ± 0.536 | 57.464 ± 2.067 | 135.567 ± 0.526 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.789 ± 0.024 | 1.939 ± 0.015 | 3.159 ± 0.035 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.462 ± 0.036 | 9.913 ± 0.065 | 18.956 ± 0.555 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 50.159 ± 3.571 | 54.754 ± 1.653 | 138.116 ± 0.214 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.835 ± 0.019 | 1.844 ± 0.012 | 3.145 ± 0.020 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.656 ± 0.045 | 9.937 ± 0.402 | 19.660 ± 0.075 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 4.738 ± 0.018 | 6.564 ± 0.188 | 5.205 ± 0.095 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.215 ± 0.003 | 0.299 ± 0.020 | 0.221 ± 0.004 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 0.988 ± 0.015 | 1.194 ± 0.051 | 1.078 ± 0.058 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 4.579 ± 0.041 | 11.059 ± 0.628 | 4.777 ± 0.038 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.210 ± 0.003 | 0.385 ± 0.010 | 0.220 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 0.944 ± 0.018 | 1.897 ± 0.058 | 0.991 ± 0.047 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 117.060 ± 2.165 | 148.652 ± 21.004 | 181.425 ± 18.767 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.051 ± 0.046 | 5.511 ± 0.063 | 6.137 ± 0.068 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 22.928 ± 0.166 | 26.399 ± 0.828 | 31.786 ± 0.213 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 119.164 ± 4.613 | 129.121 ± 20.119 | 177.752 ± 1.588 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.003 ± 0.022 | 4.955 ± 0.016 | 6.040 ± 0.031 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 22.794 ± 0.107 | 23.328 ± 0.510 | 30.834 ± 0.071 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.050 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.022 ± 0.001 | 0.033 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.065 ± 0.002 | 0.073 ± 0.002 | 0.044 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.023 ± 0.001 | 0.034 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.031 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.044 ± 0.000 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.064 ± 0.001 | 0.068 ± 0.001 | 0.045 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.023 ± 0.001 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.026 ± 0.001 | 0.037 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.033 ± 0.000 | 0.059 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.024 ± 0.001 | 0.049 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.039 ± 0.000 | 0.075 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.001 | 0.050 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.057 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.033 ± 0.000 | 0.047 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.038 ± 0.000 | 0.057 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.031 ± 0.000 | 0.046 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.050 ± 0.001 | 0.047 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.039 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.065 ± 0.001 | 0.055 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.040 ± 0.001 | 0.038 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.040 ± 0.001 | 0.045 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.050 ± 0.001 | 0.047 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.040 ± 0.000 | 0.036 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.066 ± 0.000 | 0.054 ± 0.001 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.000 | 0.044 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.017 ± 0.000 | 0.081 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.080 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.016 ± 0.000 | 0.080 ± 0.004 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.080 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.016 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.036 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.023 ± 0.000 | 0.050 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.040 ± 0.000 | 0.073 ± 0.001 | 0.024 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.080 ± 0.001 | 0.123 ± 0.002 | 0.070 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.015 ± 0.000 | 0.048 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.025 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.080 ± 0.001 | 0.102 ± 0.001 | 0.073 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_175502/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_175502`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_175502`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_175502/cpu_ops_t4_20260923_175502.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_175502/linalg_jvp_vjp_t4_20260923_175502.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 71.233 ± 2.032 | 85.335 ± 20.792 | 84.465 ± 0.425 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 2.969 ± 0.027 | 3.048 ± 0.025 | 3.071 ± 0.031 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.510 ± 0.164 | 15.769 ± 1.132 | 15.136 ± 0.212 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 70.409 ± 2.052 | 71.667 ± 0.989 | 84.281 ± 0.250 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 2.938 ± 0.034 | 2.806 ± 0.016 | 3.025 ± 0.031 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.072 ± 1.119 | 13.177 ± 0.911 | 15.121 ± 0.146 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 27.330 ± 0.741 | 71.491 ± 20.395 | 34.819 ± 0.280 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.709 ± 0.017 | 1.343 ± 0.012 | 0.760 ± 0.009 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.313 ± 0.114 | 7.720 ± 0.277 | 5.370 ± 0.163 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 28.593 ± 1.309 | 30.614 ± 1.260 | 35.799 ± 1.600 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.712 ± 0.007 | 0.876 ± 0.294 | 0.805 ± 0.012 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.560 ± 0.345 | 4.251 ± 0.068 | 5.445 ± 0.082 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 45.781 ± 0.747 | 51.149 ± 0.374 | 58.939 ± 1.235 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.842 ± 0.028 | 1.847 ± 0.020 | 1.804 ± 0.081 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.148 ± 0.640 | 9.190 ± 0.692 | 10.118 ± 0.215 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 46.771 ± 0.987 | 50.995 ± 10.517 | 63.035 ± 0.412 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.893 ± 0.030 | 1.785 ± 0.019 | 1.875 ± 0.009 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.351 ± 0.238 | 8.744 ± 0.796 | 10.308 ± 0.117 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.021 ± 0.143 | 5.250 ± 0.064 | 5.215 ± 0.242 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.237 ± 0.003 | 0.318 ± 0.038 | 0.218 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.085 ± 0.142 | 1.056 ± 0.050 | 1.150 ± 0.015 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.081 ± 0.087 | 8.856 ± 0.115 | 4.841 ± 0.161 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.224 ± 0.005 | 0.447 ± 0.041 | 0.213 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.010 ± 0.080 | 1.716 ± 0.018 | 1.102 ± 0.050 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 133.188 ± 5.835 | 136.748 ± 18.239 | 127.161 ± 1.838 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.167 ± 0.093 | 6.067 ± 0.247 | 5.287 ± 0.031 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 25.994 ± 0.533 | 25.177 ± 0.265 | 25.389 ± 1.316 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 122.447 ± 12.308 | 113.170 ± 1.084 | 125.712 ± 1.869 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.099 ± 0.073 | 5.086 ± 0.132 | 5.233 ± 0.035 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 26.112 ± 1.561 | 22.682 ± 0.366 | 24.261 ± 0.142 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.037 ± 0.001 | 0.050 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.039 ± 0.004 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.065 ± 0.001 | 0.073 ± 0.001 | 0.044 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.035 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.028 ± 0.002 | 0.042 ± 0.001 | 0.006 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.038 ± 0.001 | 0.044 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.024 ± 0.001 | 0.030 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.064 ± 0.001 | 0.070 ± 0.002 | 0.044 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.026 ± 0.000 | 0.036 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.032 ± 0.000 | 0.140 ± 0.003 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.025 ± 0.000 | 0.130 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.039 ± 0.000 | 0.143 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.001 | 0.130 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.026 ± 0.000 | 0.134 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.032 ± 0.000 | 0.103 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.025 ± 0.000 | 0.095 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.039 ± 0.000 | 0.104 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.025 ± 0.000 | 0.097 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.030 ± 0.000 | 0.099 ± 0.001 | 0.005 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.051 ± 0.001 | 0.077 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.041 ± 0.001 | 0.073 ± 0.004 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.061 ± 0.002 | 0.082 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.041 ± 0.001 | 0.073 ± 0.003 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.042 ± 0.001 | 0.078 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.051 ± 0.001 | 0.076 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.042 ± 0.001 | 0.078 ± 0.005 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.062 ± 0.001 | 0.086 ± 0.002 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.042 ± 0.001 | 0.073 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.047 ± 0.001 | 0.075 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.017 ± 0.000 | 0.086 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.017 ± 0.001 | 0.084 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.017 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.017 ± 0.000 | 0.083 ± 0.002 | 0.006 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.017 ± 0.000 | 0.040 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.043 ± 0.007 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.023 ± 0.000 | 0.050 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.016 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.039 ± 0.001 | 0.076 ± 0.003 | 0.027 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.015 ± 0.001 | 0.048 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.082 ± 0.004 | 0.123 ± 0.002 | 0.076 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.015 ± 0.000 | 0.051 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.038 ± 0.001 | 0.056 ± 0.003 | 0.027 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.034 ± 0.004 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.083 ± 0.001 | 0.102 ± 0.002 | 0.075 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.014 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.019 ± 0.000 | 0.037 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

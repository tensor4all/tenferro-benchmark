# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`
- Benchmark commit: `64970fd383409685da2dbf874a4ad9dcdff4fec0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_162342/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260916_162342`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260916_162342`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_162342/cpu_ops_t1_20260916_162342.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_162342/linalg_jvp_vjp_t1_20260916_162342.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 75.552 ± 0.239 | 81.199 ± 0.603 | 139.174 ± 2.688 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.974 ± 0.041 | 3.016 ± 0.076 | 3.999 ± 0.023 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.923 ± 0.073 | 14.525 ± 0.127 | 21.997 ± 0.199 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 79.554 ± 0.993 | 69.656 ± 0.418 | 138.078 ± 11.930 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.953 ± 0.021 | 2.781 ± 0.025 | 3.960 ± 0.035 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.864 ± 0.099 | 12.892 ± 0.197 | 21.679 ± 0.220 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 31.801 ± 1.450 | 62.799 ± 19.106 | 87.654 ± 0.179 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.684 ± 0.006 | 1.316 ± 0.025 | 1.744 ± 0.020 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 4.185 ± 0.086 | 8.645 ± 0.118 | 12.279 ± 0.127 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 31.390 ± 1.140 | 32.842 ± 0.683 | 89.454 ± 0.867 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.704 ± 0.008 | 0.779 ± 0.019 | 1.747 ± 0.021 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.338 ± 0.107 | 4.740 ± 0.091 | 12.254 ± 0.173 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 51.245 ± 0.493 | 57.627 ± 0.472 | 137.206 ± 2.573 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.880 ± 0.021 | 1.938 ± 0.024 | 3.195 ± 0.020 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.751 ± 0.060 | 10.072 ± 0.100 | 19.231 ± 0.179 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 51.363 ± 0.579 | 56.459 ± 6.403 | 144.051 ± 7.900 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.882 ± 0.022 | 1.852 ± 0.021 | 3.171 ± 0.021 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.985 ± 0.091 | 9.624 ± 0.154 | 19.966 ± 2.015 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 6.032 ± 0.310 | 5.847 ± 0.082 | 5.591 ± 0.262 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.292 ± 0.003 | 0.293 ± 0.023 | 0.219 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.251 ± 0.045 | 1.203 ± 0.051 | 1.039 ± 0.028 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 6.030 ± 0.114 | 9.804 ± 0.124 | 5.274 ± 0.407 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.287 ± 0.004 | 0.388 ± 0.008 | 0.217 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.197 ± 0.055 | 1.887 ± 0.085 | 0.996 ± 0.031 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 118.460 ± 0.235 | 137.047 ± 12.934 | 180.814 ± 2.313 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.065 ± 0.050 | 5.539 ± 0.029 | 6.154 ± 0.039 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 23.312 ± 0.133 | 26.189 ± 0.100 | 31.421 ± 0.299 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 117.921 ± 0.442 | 113.731 ± 0.414 | 179.795 ± 15.460 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.043 ± 0.056 | 5.006 ± 0.045 | 6.111 ± 0.041 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 23.293 ± 0.409 | 22.862 ± 0.122 | 33.517 ± 2.970 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.042 ± 0.000 | 0.051 ± 0.002 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.026 ± 0.000 | 0.034 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.073 ± 0.001 | 0.077 ± 0.001 | 0.045 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.028 ± 0.007 | 0.035 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.033 ± 0.003 | 0.043 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.041 ± 0.001 | 0.045 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.025 ± 0.000 | 0.029 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.066 ± 0.000 | 0.071 ± 0.002 | 0.046 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.025 ± 0.000 | 0.030 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.033 ± 0.001 | 0.037 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.040 ± 0.000 | 0.062 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.000 | 0.051 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.050 ± 0.000 | 0.075 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.031 ± 0.000 | 0.051 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.031 ± 0.001 | 0.058 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.040 ± 0.001 | 0.048 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.031 ± 0.000 | 0.041 ± 0.013 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.054 ± 0.009 | 0.057 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.031 ± 0.000 | 0.040 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.038 ± 0.001 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.064 ± 0.000 | 0.047 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.048 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.075 ± 0.001 | 0.056 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.048 ± 0.001 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.049 ± 0.000 | 0.045 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.063 ± 0.001 | 0.047 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.049 ± 0.001 | 0.038 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.075 ± 0.000 | 0.055 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.049 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.056 ± 0.001 | 0.044 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.036 ± 0.000 | 0.085 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.034 ± 0.000 | 0.086 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.046 ± 0.002 | 0.093 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.034 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.035 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.035 ± 0.001 | 0.039 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.044 ± 0.000 | 0.050 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.033 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.033 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.044 ± 0.001 | 0.076 ± 0.003 | 0.026 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.018 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.086 ± 0.002 | 0.125 ± 0.004 | 0.074 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.019 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.028 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.043 ± 0.000 | 0.057 ± 0.001 | 0.027 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.016 ± 0.000 | 0.031 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.085 ± 0.002 | 0.103 ± 0.003 | 0.075 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.017 ± 0.000 | 0.032 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.022 ± 0.000 | 0.036 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_164245/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260916_164245`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260916_164245`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_164245/cpu_ops_t4_20260916_164245.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_164245/linalg_jvp_vjp_t4_20260916_164245.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 73.377 ± 0.873 | 79.092 ± 0.458 | 84.400 ± 1.222 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 2.966 ± 0.025 | 2.970 ± 0.023 | 3.066 ± 0.058 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.280 ± 0.090 | 14.135 ± 0.165 | 15.106 ± 0.110 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 73.753 ± 0.173 | 68.030 ± 0.238 | 85.039 ± 3.362 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 2.937 ± 0.016 | 2.720 ± 0.027 | 3.073 ± 0.045 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.705 ± 0.354 | 12.528 ± 0.072 | 15.075 ± 0.062 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 27.751 ± 0.376 | 54.281 ± 1.146 | 35.211 ± 0.636 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.762 ± 0.008 | 1.341 ± 0.015 | 0.760 ± 0.009 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.191 ± 0.044 | 7.565 ± 0.103 | 5.442 ± 0.179 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 28.158 ± 0.395 | 29.437 ± 0.215 | 36.130 ± 0.330 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.739 ± 0.010 | 0.851 ± 0.006 | 0.809 ± 0.014 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.161 ± 0.020 | 4.193 ± 0.023 | 5.453 ± 0.077 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 47.134 ± 0.938 | 51.033 ± 0.736 | 58.636 ± 1.404 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.937 ± 0.018 | 1.860 ± 0.029 | 1.821 ± 0.068 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.467 ± 0.328 | 8.362 ± 0.065 | 9.985 ± 0.654 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 47.300 ± 2.675 | 50.427 ± 1.784 | 63.491 ± 0.467 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.906 ± 0.027 | 1.783 ± 0.018 | 1.880 ± 0.015 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.253 ± 0.140 | 8.119 ± 0.054 | 10.273 ± 0.078 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.375 ± 0.042 | 5.160 ± 0.065 | 5.042 ± 0.100 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.326 ± 0.004 | 0.291 ± 0.009 | 0.219 ± 0.004 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.209 ± 0.072 | 1.033 ± 0.009 | 1.100 ± 0.052 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.307 ± 0.285 | 8.758 ± 0.046 | 4.839 ± 0.259 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.317 ± 0.005 | 0.407 ± 0.003 | 0.212 ± 0.004 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.167 ± 0.009 | 1.718 ± 0.018 | 1.025 ± 0.073 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 116.890 ± 2.352 | 131.926 ± 0.546 | 128.462 ± 0.898 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.023 ± 0.033 | 5.574 ± 0.096 | 5.289 ± 0.031 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.422 ± 0.572 | 25.003 ± 0.181 | 24.338 ± 0.094 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 115.795 ± 0.397 | 111.717 ± 0.399 | 127.481 ± 1.105 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 4.980 ± 0.018 | 4.937 ± 0.045 | 5.226 ± 0.031 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.374 ± 0.252 | 22.390 ± 0.043 | 24.206 ± 0.073 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.042 ± 0.001 | 0.050 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.025 ± 0.000 | 0.033 ± 0.000 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.072 ± 0.001 | 0.073 ± 0.002 | 0.044 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.026 ± 0.000 | 0.034 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.033 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.038 ± 0.004 | 0.042 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.066 ± 0.000 | 0.066 ± 0.001 | 0.044 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.040 ± 0.001 | 0.142 ± 0.006 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.031 ± 0.000 | 0.128 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.052 ± 0.001 | 0.148 ± 0.009 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.031 ± 0.000 | 0.129 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.031 ± 0.000 | 0.132 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.039 ± 0.000 | 0.104 ± 0.010 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.031 ± 0.000 | 0.094 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.054 ± 0.008 | 0.107 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.031 ± 0.001 | 0.095 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.037 ± 0.000 | 0.099 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.063 ± 0.001 | 0.075 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.049 ± 0.001 | 0.068 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.076 ± 0.000 | 0.082 ± 0.006 | 0.020 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.048 ± 0.000 | 0.071 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.049 ± 0.001 | 0.073 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.062 ± 0.001 | 0.075 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.049 ± 0.001 | 0.074 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.075 ± 0.000 | 0.086 ± 0.017 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.049 ± 0.000 | 0.073 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.056 ± 0.001 | 0.075 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.036 ± 0.001 | 0.086 ± 0.006 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.034 ± 0.000 | 0.081 ± 0.002 | 0.003 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.048 ± 0.001 | 0.091 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.034 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.035 ± 0.000 | 0.082 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.034 ± 0.001 | 0.040 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.043 ± 0.002 | 0.051 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.034 ± 0.001 | 0.038 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.043 ± 0.000 | 0.078 ± 0.003 | 0.026 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.018 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.088 ± 0.002 | 0.122 ± 0.003 | 0.073 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.019 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.028 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.042 ± 0.000 | 0.057 ± 0.002 | 0.028 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.016 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.087 ± 0.001 | 0.098 ± 0.004 | 0.075 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.017 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.021 ± 0.000 | 0.036 ± 0.001 | 0.008 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`
- Benchmark commit: `5df994b8aee8e926b52e885ab89a5732e4e2ff8f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_201832/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_201832`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_201832`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_201832/cpu_ops_t1_20260923_201832.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_201832/linalg_jvp_vjp_t1_20260923_201832.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 71.144 ± 0.291 | 80.812 ± 0.601 | 134.101 ± 0.255 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.922 ± 0.031 | 3.051 ± 0.019 | 3.897 ± 0.015 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.298 ± 0.055 | 14.471 ± 0.098 | 21.305 ± 0.062 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 70.883 ± 0.762 | 69.317 ± 0.848 | 132.853 ± 0.189 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.876 ± 0.008 | 2.798 ± 0.023 | 3.884 ± 0.018 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.193 ± 0.052 | 12.916 ± 0.047 | 21.123 ± 0.126 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 32.001 ± 1.278 | 59.123 ± 0.989 | 85.911 ± 0.266 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.656 ± 0.003 | 1.295 ± 0.020 | 1.713 ± 0.005 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 3.983 ± 0.028 | 8.359 ± 0.037 | 11.859 ± 0.067 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 28.797 ± 0.903 | 30.872 ± 0.674 | 86.545 ± 0.784 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.678 ± 0.005 | 0.763 ± 0.007 | 1.722 ± 0.016 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.002 ± 0.075 | 4.585 ± 0.067 | 11.712 ± 0.045 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 49.817 ± 0.552 | 56.094 ± 0.538 | 134.247 ± 0.251 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.787 ± 0.010 | 1.909 ± 0.039 | 3.145 ± 0.021 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.538 ± 0.037 | 9.843 ± 0.055 | 18.737 ± 0.059 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 50.146 ± 0.598 | 53.597 ± 1.955 | 136.868 ± 3.204 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.829 ± 0.009 | 1.834 ± 0.011 | 3.122 ± 0.012 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.708 ± 0.019 | 9.324 ± 0.068 | 19.406 ± 0.083 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 4.753 ± 0.050 | 5.611 ± 0.085 | 5.165 ± 0.054 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.216 ± 0.011 | 0.293 ± 0.008 | 0.220 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.000 ± 0.045 | 1.182 ± 0.046 | 1.019 ± 0.016 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 4.629 ± 0.074 | 9.263 ± 0.041 | 4.752 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.209 ± 0.001 | 0.387 ± 0.006 | 0.215 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 0.942 ± 0.016 | 1.862 ± 0.010 | 0.978 ± 0.013 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 116.999 ± 0.950 | 133.649 ± 1.495 | 176.614 ± 0.626 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.012 ± 0.036 | 5.563 ± 0.094 | 6.051 ± 0.024 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 22.796 ± 0.103 | 25.756 ± 0.142 | 30.771 ± 0.044 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 115.746 ± 0.731 | 112.015 ± 1.212 | 175.465 ± 0.389 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.009 ± 0.029 | 5.054 ± 0.063 | 6.009 ± 0.024 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 22.843 ± 0.096 | 22.482 ± 0.149 | 30.517 ± 0.051 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.040 ± 0.000 | 0.050 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.022 ± 0.001 | 0.033 ± 0.001 | 0.003 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.064 ± 0.001 | 0.074 ± 0.001 | 0.045 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.032 ± 0.000 | 0.043 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.045 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.063 ± 0.000 | 0.068 ± 0.001 | 0.045 ± 0.003 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.025 ± 0.001 | 0.037 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.033 ± 0.000 | 0.061 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.038 ± 0.000 | 0.075 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.024 ± 0.000 | 0.057 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.032 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.038 ± 0.000 | 0.056 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.023 ± 0.001 | 0.039 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.031 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.050 ± 0.000 | 0.048 ± 0.001 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.064 ± 0.000 | 0.054 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.040 ± 0.001 | 0.046 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.050 ± 0.000 | 0.047 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.040 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.065 ± 0.001 | 0.054 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.040 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.000 | 0.045 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.017 ± 0.000 | 0.082 ± 0.004 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.081 ± 0.005 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.016 ± 0.000 | 0.038 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.023 ± 0.000 | 0.049 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.040 ± 0.000 | 0.075 ± 0.001 | 0.027 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.079 ± 0.001 | 0.122 ± 0.001 | 0.074 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.015 ± 0.000 | 0.048 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.022 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.026 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.078 ± 0.002 | 0.102 ± 0.001 | 0.075 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.018 ± 0.001 | 0.035 ± 0.002 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_204226/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260923_204226`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260923_204226`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_204226/cpu_ops_t4_20260923_204226.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_204226/linalg_jvp_vjp_t4_20260923_204226.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 68.940 ± 1.360 | 87.304 ± 4.373 | 86.684 ± 2.318 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 2.922 ± 0.031 | 3.095 ± 0.058 | 3.094 ± 0.052 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 12.914 ± 0.074 | 14.453 ± 0.179 | 15.311 ± 0.157 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 68.503 ± 0.468 | 72.967 ± 1.888 | 87.198 ± 4.028 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 2.898 ± 0.018 | 2.833 ± 0.083 | 3.040 ± 0.019 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 12.811 ± 0.050 | 12.822 ± 0.348 | 15.126 ± 0.211 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 26.616 ± 0.251 | 70.549 ± 11.496 | 36.209 ± 1.115 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.702 ± 0.003 | 1.402 ± 0.100 | 0.770 ± 0.024 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 3.926 ± 0.022 | 8.110 ± 0.683 | 5.382 ± 0.152 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 27.160 ± 0.236 | 31.979 ± 4.268 | 36.525 ± 0.775 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.708 ± 0.007 | 0.863 ± 0.039 | 0.828 ± 0.026 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.038 ± 0.034 | 4.401 ± 0.310 | 5.475 ± 0.123 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 51.812 ± 0.964 | 53.852 ± 2.599 | 73.126 ± 13.614 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.819 ± 0.006 | 1.892 ± 0.101 | 1.854 ± 0.144 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 7.881 ± 0.024 | 8.647 ± 0.191 | 10.108 ± 0.192 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 51.117 ± 1.591 | 53.234 ± 4.133 | 66.130 ± 1.675 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.829 ± 0.016 | 1.917 ± 0.135 | 1.904 ± 0.062 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 7.997 ± 0.060 | 8.543 ± 0.268 | 10.383 ± 0.152 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 4.549 ± 0.061 | 5.129 ± 0.058 | 4.955 ± 0.194 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.236 ± 0.005 | 0.296 ± 0.010 | 0.230 ± 0.019 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 0.956 ± 0.008 | 1.079 ± 0.082 | 1.143 ± 0.029 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 4.447 ± 0.037 | 8.695 ± 0.326 | 4.692 ± 0.099 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.224 ± 0.001 | 0.407 ± 0.010 | 0.216 ± 0.023 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 0.933 ± 0.011 | 1.782 ± 0.082 | 1.086 ± 0.011 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 131.924 ± 32.161 | 137.546 ± 18.473 | 130.580 ± 6.297 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 4.994 ± 0.027 | 6.106 ± 0.239 | 5.333 ± 0.082 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 22.825 ± 0.028 | 25.802 ± 0.511 | 25.050 ± 0.820 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 120.989 ± 5.414 | 117.604 ± 3.132 | 132.051 ± 6.428 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 4.973 ± 0.020 | 5.396 ± 0.207 | 5.386 ± 0.263 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 22.762 ± 0.065 | 23.147 ± 0.276 | 24.504 ± 0.706 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.040 ± 0.000 | 0.051 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.065 ± 0.001 | 0.078 ± 0.003 | 0.044 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.023 ± 0.000 | 0.034 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.032 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.039 ± 0.000 | 0.043 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.064 ± 0.000 | 0.071 ± 0.002 | 0.044 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.033 ± 0.000 | 0.142 ± 0.010 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.131 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.039 ± 0.001 | 0.146 ± 0.008 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.130 ± 0.005 | 0.004 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.000 | 0.137 ± 0.003 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.033 ± 0.000 | 0.106 ± 0.008 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.024 ± 0.000 | 0.093 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.038 ± 0.000 | 0.114 ± 0.009 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.093 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.032 ± 0.000 | 0.099 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.051 ± 0.001 | 0.077 ± 0.007 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.040 ± 0.000 | 0.072 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.065 ± 0.001 | 0.084 ± 0.003 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.040 ± 0.000 | 0.072 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.041 ± 0.001 | 0.074 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.051 ± 0.002 | 0.085 ± 0.013 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.041 ± 0.001 | 0.072 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.066 ± 0.000 | 0.091 ± 0.011 | 0.019 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.040 ± 0.001 | 0.071 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.048 ± 0.002 | 0.073 ± 0.003 | 0.005 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.017 ± 0.000 | 0.086 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.016 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.024 ± 0.000 | 0.095 ± 0.008 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.016 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.082 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.016 ± 0.000 | 0.040 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.015 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.023 ± 0.000 | 0.051 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.016 ± 0.000 | 0.037 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.040 ± 0.000 | 0.075 ± 0.002 | 0.028 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.080 ± 0.001 | 0.125 ± 0.004 | 0.075 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.015 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.028 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.000 | 0.031 ± 0.002 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.081 ± 0.003 | 0.103 ± 0.004 | 0.076 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.014 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.018 ± 0.000 | 0.036 ± 0.001 | 0.009 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

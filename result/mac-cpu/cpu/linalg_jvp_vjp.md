# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`
- Benchmark commit: `cc2153b9548e1542f955c011dccfb77a98196fed`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261023_172845/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261023_172845`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261023_172845`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261023_172845/cpu_ops_t1_20261023_172845.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261023_172845/linalg_jvp_vjp_t1_20261023_172845.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 5.086 ± 0.202 | 5.635 ± 0.398 | 8.242 ± 1.671 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 31.257 ± 6.275 | 29.048 ± 3.498 | 41.052 ± 0.527 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.630 ± 1.807 | 4.426 ± 0.049 | 7.449 ± 2.113 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 30.050 ± 1.592 | 27.937 ± 0.949 | 40.756 ± 1.094 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 1.098 ± 0.168 | 1.743 ± 0.022 | 3.407 ± 1.567 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 7.730 ± 0.092 | 14.935 ± 1.590 | 20.210 ± 0.429 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 1.106 ± 0.076 | 1.061 ± 0.015 | 2.644 ± 0.124 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 8.115 ± 0.244 | 7.611 ± 0.132 | 20.319 ± 2.853 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 2.405 ± 0.129 | 2.788 ± 0.189 | 5.081 ± 2.403 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 14.045 ± 0.368 | 15.647 ± 1.253 | 31.754 ± 2.389 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 2.406 ± 0.221 | 2.666 ± 0.243 | 4.850 ± 3.048 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 13.886 ± 2.280 | 13.595 ± 3.433 | 31.104 ± 3.932 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.286 ± 0.243 | 0.478 ± 0.077 | 0.321 ± 0.006 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.855 ± 2.819 | 1.828 ± 0.105 | 2.196 ± 2.998 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.285 ± 0.006 | 0.570 ± 0.013 | 0.342 ± 0.016 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.561 ± 0.064 | 2.615 ± 0.191 | 1.859 ± 0.232 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 9.147 ± 0.036 | 10.290 ± 0.467 | 10.856 ± 0.140 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 57.114 ± 4.927 | 59.378 ± 12.848 | 67.015 ± 1.660 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 9.147 ± 0.178 | 9.358 ± 0.389 | 11.783 ± 3.825 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 56.743 ± 5.610 | 52.934 ± 4.382 | 67.577 ± 2.266 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.015 ± 0.004 | 0.067 ± 0.002 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.017 ± 0.002 | 0.070 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.025 ± 0.000 | 0.081 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.018 ± 0.006 | 0.058 ± 0.001 | 0.007 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.017 ± 0.000 | 0.059 ± 0.003 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.023 ± 0.010 | 0.073 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.018 ± 0.013 | 0.100 ± 0.004 | 0.008 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.018 ± 0.001 | 0.104 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.017 ± 0.000 | 0.123 ± 0.017 | 0.010 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.020 ± 0.003 | 0.079 ± 0.011 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.079 ± 0.009 | 0.010 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.020 ± 0.000 | 0.089 ± 0.003 | 0.010 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.008 | 0.078 ± 0.024 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.029 ± 0.006 | 0.076 ± 0.003 | 0.008 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.029 ± 0.036 | 0.087 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.032 ± 0.007 | 0.075 ± 0.001 | 0.008 ± 0.007 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.035 ± 0.012 | 0.076 ± 0.007 | 0.008 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.033 ± 0.003 | 0.085 ± 0.001 | 0.011 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.016 ± 0.010 | 0.163 ± 0.027 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.157 ± 0.010 | 0.008 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.012 ± 0.003 | 0.155 ± 0.004 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.012 ± 0.001 | 0.076 ± 0.001 | 0.009 ± 0.003 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.012 ± 0.000 | 0.076 ± 0.002 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.012 ± 0.001 | 0.077 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.012 ± 0.013 | 0.108 ± 0.030 | 0.013 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.014 ± 0.002 | 0.097 ± 0.009 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.026 ± 0.011 | 0.114 ± 0.008 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.013 ± 0.002 | 0.063 ± 0.002 | 0.014 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.013 ± 0.003 | 0.064 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.020 ± 0.001 | 0.075 ± 0.007 | 0.018 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261023_175432/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261023_175432`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261023_175432`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261023_175432/cpu_ops_t4_20261023_175432.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261023_175432/linalg_jvp_vjp_t4_20261023_175432.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 4.926 ± 0.491 | 6.023 ± 0.562 | 7.485 ± 3.072 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 28.857 ± 0.600 | 36.245 ± 16.585 | 34.556 ± 3.478 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.070 ± 0.461 | 5.403 ± 0.153 | 7.023 ± 1.090 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.014 ± 1.274 | 30.686 ± 4.866 | 34.637 ± 2.313 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.682 ± 0.295 | 3.297 ± 0.348 | 2.998 ± 1.491 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 9.587 ± 1.285 | 19.020 ± 4.205 | 10.956 ± 3.369 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.261 ± 0.243 | 2.364 ± 0.816 | 2.656 ± 1.611 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 7.940 ± 0.319 | 20.914 ± 9.351 | 11.082 ± 3.736 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 4.106 ± 1.868 | 3.812 ± 0.650 | 3.728 ± 1.858 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 15.296 ± 0.949 | 16.797 ± 1.347 | 24.850 ± 7.190 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.876 ± 0.298 | 4.071 ± 4.410 | 4.625 ± 1.596 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 17.475 ± 2.718 | 16.151 ± 5.793 | 25.101 ± 3.284 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.328 ± 0.033 | 0.720 ± 0.223 | 0.354 ± 0.021 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.384 ± 0.015 | 2.348 ± 0.556 | 2.884 ± 4.100 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.508 ± 0.040 | 0.741 ± 0.124 | 0.524 ± 0.194 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.563 ± 0.281 | 4.933 ± 2.351 | 2.274 ± 1.532 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 9.548 ± 0.552 | 11.924 ± 0.828 | 10.498 ± 0.582 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 62.686 ± 10.217 | 65.097 ± 13.239 | 71.775 ± 7.913 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 9.663 ± 0.285 | 9.634 ± 0.613 | 10.925 ± 1.046 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 71.659 ± 2.771 | 55.223 ± 6.573 | 77.494 ± 24.121 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.015 ± 0.001 | 0.121 ± 0.065 | 0.008 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.016 ± 0.001 | 0.103 ± 0.087 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.022 ± 0.001 | 0.242 ± 0.072 | 0.010 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.071 ± 0.021 | 0.008 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.015 ± 0.001 | 0.113 ± 0.028 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.029 ± 0.012 | 0.235 ± 0.194 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.017 ± 0.002 | 0.521 ± 0.229 | 0.010 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.015 ± 0.000 | 0.409 ± 0.085 | 0.018 ± 0.011 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.016 ± 0.003 | 0.484 ± 0.069 | 0.006 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.015 ± 0.000 | 0.457 ± 0.169 | 0.007 ± 0.014 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.014 ± 0.001 | 0.332 ± 0.220 | 0.011 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.019 ± 0.002 | 0.421 ± 0.176 | 0.019 ± 0.012 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.030 ± 0.004 | 0.243 ± 0.028 | 0.010 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.024 ± 0.004 | 0.294 ± 0.241 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.034 ± 0.033 | 0.293 ± 0.288 | 0.006 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.026 ± 0.001 | 0.263 ± 0.037 | 0.008 ± 0.006 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.025 ± 0.002 | 0.257 ± 0.086 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.030 ± 0.002 | 0.311 ± 0.175 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.012 ± 0.001 | 0.212 ± 0.248 | 0.008 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.010 ± 0.000 | 0.192 ± 0.032 | 0.009 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.021 ± 0.004 | 0.201 ± 0.739 | 0.013 ± 0.006 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.011 ± 0.000 | 0.092 ± 0.092 | 0.008 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.102 ± 0.017 | 0.009 ± 0.004 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.013 ± 0.000 | 0.087 ± 0.086 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.010 ± 0.001 | 0.211 ± 0.077 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.012 ± 0.001 | 0.247 ± 0.117 | 0.013 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.003 | 0.184 ± 0.074 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.011 ± 0.000 | 0.072 ± 0.019 | 0.013 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.012 ± 0.000 | 0.091 ± 0.021 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.027 ± 0.011 | 0.133 ± 0.085 | 0.018 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

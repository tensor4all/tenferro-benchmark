# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`
- Benchmark commit: `f983e88b9d07e558989427d047f1fa4676df2ea8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_102753/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260916_102753`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260916_102753`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_102753/cpu_ops_t1_20260916_102753.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_102753/linalg_jvp_vjp_t1_20260916_102753.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 74.347 ± 0.350 | 82.103 ± 1.133 | 136.261 ± 9.167 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.924 ± 0.043 | 3.098 ± 0.064 | 3.939 ± 0.042 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.582 ± 0.056 | 14.580 ± 0.124 | 21.513 ± 0.071 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 74.757 ± 0.219 | 70.857 ± 1.101 | 134.355 ± 0.581 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.879 ± 0.048 | 2.847 ± 0.062 | 3.922 ± 0.067 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.559 ± 0.099 | 13.091 ± 0.151 | 21.220 ± 0.100 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 28.958 ± 1.979 | 61.912 ± 1.680 | 86.958 ± 0.430 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.712 ± 0.057 | 1.315 ± 0.063 | 1.744 ± 0.047 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 4.235 ± 0.196 | 8.494 ± 0.078 | 11.950 ± 0.052 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 29.216 ± 0.327 | 32.476 ± 0.914 | 88.168 ± 0.530 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.714 ± 0.034 | 0.789 ± 0.038 | 1.758 ± 0.028 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.261 ± 0.065 | 4.676 ± 0.121 | 11.852 ± 0.075 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 50.357 ± 0.944 | 56.874 ± 1.046 | 138.353 ± 3.481 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.859 ± 0.046 | 1.948 ± 0.085 | 3.181 ± 0.037 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.865 ± 0.057 | 10.020 ± 0.112 | 19.141 ± 0.171 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 50.690 ± 0.267 | 56.327 ± 1.825 | 138.443 ± 0.422 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.853 ± 0.039 | 1.856 ± 0.079 | 3.158 ± 0.046 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.855 ± 0.037 | 9.462 ± 0.068 | 19.586 ± 0.150 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 5.769 ± 0.145 | 6.054 ± 0.382 | 5.562 ± 0.235 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.297 ± 0.012 | 0.299 ± 0.028 | 0.244 ± 0.010 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.237 ± 0.054 | 1.207 ± 0.105 | 1.018 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 5.757 ± 0.150 | 9.719 ± 0.135 | 4.887 ± 0.300 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.287 ± 0.010 | 0.389 ± 0.013 | 0.238 ± 0.009 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.192 ± 0.042 | 1.947 ± 0.059 | 0.976 ± 0.014 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 117.975 ± 0.321 | 137.658 ± 2.976 | 179.347 ± 0.818 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.036 ± 0.024 | 5.530 ± 0.031 | 6.105 ± 0.043 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 23.213 ± 0.101 | 26.135 ± 0.188 | 31.155 ± 0.101 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 117.264 ± 0.571 | 114.222 ± 1.237 | 179.820 ± 3.678 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.030 ± 0.050 | 5.056 ± 0.070 | 6.054 ± 0.049 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 23.153 ± 0.114 | 22.765 ± 0.091 | 30.993 ± 0.110 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_103859/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260916_103859`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260916_103859`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_103859/cpu_ops_t4_20260916_103859.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_103859/linalg_jvp_vjp_t4_20260916_103859.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 76.286 ± 8.273 | 80.619 ± 0.689 | 85.989 ± 0.693 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.021 ± 0.026 | 3.372 ± 0.241 | 3.122 ± 0.062 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.266 ± 1.104 | 14.836 ± 0.087 | 15.370 ± 0.100 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 74.391 ± 4.684 | 71.138 ± 4.576 | 86.042 ± 0.278 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 2.990 ± 0.018 | 2.988 ± 0.375 | 3.088 ± 0.082 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.410 ± 0.111 | 13.362 ± 0.216 | 15.367 ± 0.099 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 31.104 ± 0.957 | 67.921 ± 20.734 | 36.060 ± 0.345 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.912 ± 0.064 | 1.389 ± 0.126 | 0.793 ± 0.041 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.386 ± 0.221 | 8.237 ± 0.426 | 5.477 ± 0.200 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 30.232 ± 0.488 | 32.958 ± 1.423 | 36.520 ± 0.550 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.866 ± 0.019 | 0.867 ± 0.081 | 0.837 ± 0.060 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.738 ± 2.670 | 4.556 ± 0.092 | 5.775 ± 0.486 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 49.491 ± 0.993 | 58.586 ± 16.150 | 60.426 ± 0.452 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.035 ± 0.044 | 1.871 ± 0.042 | 1.807 ± 0.112 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.735 ± 0.144 | 8.923 ± 0.106 | 10.191 ± 0.357 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 52.081 ± 3.514 | 61.045 ± 17.880 | 64.384 ± 0.291 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.986 ± 0.015 | 1.804 ± 0.073 | 1.929 ± 0.031 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.349 ± 0.099 | 8.759 ± 0.213 | 10.427 ± 0.120 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 6.144 ± 0.387 | 5.787 ± 0.178 | 5.348 ± 0.371 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.409 ± 0.028 | 0.294 ± 0.021 | 0.237 ± 0.018 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.700 ± 0.164 | 1.120 ± 0.123 | 1.250 ± 0.129 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.972 ± 0.184 | 9.586 ± 0.388 | 5.032 ± 0.330 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.372 ± 0.012 | 0.406 ± 0.050 | 0.231 ± 0.014 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.396 ± 0.086 | 1.781 ± 0.090 | 1.087 ± 0.122 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 123.011 ± 10.576 | 133.014 ± 2.689 | 129.844 ± 1.382 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.123 ± 0.032 | 5.922 ± 0.516 | 5.384 ± 0.051 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.233 ± 0.188 | 26.387 ± 0.581 | 24.728 ± 0.175 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 118.477 ± 2.470 | 113.138 ± 0.576 | 129.953 ± 0.633 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.083 ± 0.023 | 5.105 ± 0.041 | 5.330 ± 0.052 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.149 ± 1.383 | 23.462 ± 0.838 | 24.532 ± 0.205 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

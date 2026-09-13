# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`
- Benchmark commit: `4d4b94aca2e2bdc19b605180e4ce524dc67f1df0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260913_182541/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_182541`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_182541`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260913_182541/cpu_ops_t1_20260913_182541.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_182541/linalg_jvp_vjp_t1_20260913_182541.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 74.522 ± 0.447 | 82.045 ± 1.380 | 136.169 ± 0.351 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 2.918 ± 0.038 | 3.054 ± 0.044 | 3.978 ± 0.031 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 13.618 ± 0.057 | 14.599 ± 0.089 | 21.716 ± 0.151 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 75.929 ± 0.809 | 70.560 ± 0.652 | 134.992 ± 0.628 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 2.890 ± 0.045 | 2.850 ± 0.061 | 3.946 ± 0.043 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 13.585 ± 0.107 | 13.228 ± 0.194 | 21.526 ± 0.152 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 28.585 ± 0.231 | 61.342 ± 2.983 | 87.655 ± 0.223 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.712 ± 0.079 | 1.330 ± 0.041 | 1.768 ± 0.041 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 4.199 ± 0.248 | 8.570 ± 0.151 | 12.154 ± 0.094 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 29.350 ± 0.515 | 32.404 ± 0.729 | 89.022 ± 0.542 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.712 ± 0.023 | 0.765 ± 0.029 | 1.758 ± 0.037 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.246 ± 0.122 | 4.724 ± 0.058 | 12.033 ± 0.088 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 50.750 ± 0.874 | 56.956 ± 0.467 | 136.926 ± 0.314 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.877 ± 0.082 | 1.938 ± 0.030 | 3.195 ± 0.031 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 8.929 ± 0.071 | 10.066 ± 0.143 | 19.224 ± 0.057 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 51.766 ± 0.384 | 55.607 ± 1.536 | 139.332 ± 1.157 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.859 ± 0.059 | 1.863 ± 0.068 | 3.174 ± 0.028 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 8.896 ± 0.036 | 9.446 ± 0.098 | 19.887 ± 0.212 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 5.847 ± 0.130 | 6.005 ± 0.216 | 5.543 ± 0.250 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.293 ± 0.004 | 0.294 ± 0.008 | 0.240 ± 0.006 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.224 ± 0.033 | 1.194 ± 0.089 | 1.025 ± 0.036 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 5.786 ± 0.115 | 9.989 ± 0.285 | 5.008 ± 0.196 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.287 ± 0.003 | 0.386 ± 0.009 | 0.238 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.194 ± 0.043 | 1.887 ± 0.035 | 0.985 ± 0.027 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 118.576 ± 0.605 | 136.280 ± 1.237 | 179.913 ± 0.692 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.070 ± 0.065 | 5.521 ± 0.082 | 6.135 ± 0.076 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 23.289 ± 0.141 | 26.079 ± 0.166 | 31.304 ± 0.173 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 117.547 ± 0.328 | 114.618 ± 1.309 | 179.788 ± 3.180 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.068 ± 0.034 | 5.077 ± 0.086 | 6.061 ± 0.038 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 23.218 ± 0.107 | 22.859 ± 0.101 | 31.205 ± 0.081 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260913_183437/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_183437`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_183437`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260913_183437/cpu_ops_t4_20260913_183437.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_183437/linalg_jvp_vjp_t4_20260913_183437.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 74.957 ± 0.446 | 81.886 ± 0.679 | 85.585 ± 0.200 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.144 ± 0.052 | 3.096 ± 0.068 | 3.077 ± 0.063 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.854 ± 0.099 | 14.372 ± 0.086 | 15.320 ± 0.180 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 75.149 ± 0.209 | 70.606 ± 0.322 | 86.138 ± 1.170 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.098 ± 0.048 | 2.861 ± 0.052 | 2.898 ± 0.065 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.805 ± 0.093 | 13.013 ± 0.215 | 15.267 ± 0.132 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 29.543 ± 0.493 | 58.529 ± 1.211 | 35.740 ± 0.482 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.915 ± 0.094 | 1.379 ± 0.076 | 0.773 ± 0.036 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.515 ± 0.198 | 7.828 ± 0.108 | 5.385 ± 0.163 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 30.085 ± 0.380 | 30.610 ± 0.109 | 36.344 ± 0.254 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.872 ± 0.038 | 0.851 ± 0.025 | 0.832 ± 0.050 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.378 ± 0.087 | 4.420 ± 0.200 | 5.498 ± 0.192 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 47.387 ± 1.201 | 52.356 ± 0.498 | 60.559 ± 1.943 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.081 ± 0.103 | 1.885 ± 0.028 | 1.805 ± 0.097 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.689 ± 0.060 | 8.710 ± 0.210 | 9.986 ± 0.512 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 47.865 ± 0.476 | 50.297 ± 0.729 | 64.171 ± 0.456 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.049 ± 0.075 | 1.829 ± 0.072 | 1.910 ± 0.033 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.440 ± 0.044 | 8.307 ± 0.106 | 10.466 ± 0.136 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.787 ± 0.144 | 5.568 ± 0.285 | 5.253 ± 0.155 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.385 ± 0.034 | 0.306 ± 0.022 | 0.207 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.267 ± 0.036 | 1.060 ± 0.095 | 1.163 ± 0.037 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.609 ± 0.181 | 9.452 ± 0.164 | 5.022 ± 0.330 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.382 ± 0.019 | 0.414 ± 0.014 | 0.201 ± 0.018 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.242 ± 0.026 | 1.739 ± 0.075 | 1.075 ± 0.040 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 119.470 ± 0.595 | 134.011 ± 0.980 | 129.422 ± 1.931 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.206 ± 0.039 | 5.620 ± 0.113 | 5.369 ± 0.073 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.483 ± 0.100 | 25.434 ± 0.064 | 24.566 ± 0.061 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 118.798 ± 0.781 | 114.242 ± 1.366 | 127.825 ± 0.553 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.168 ± 0.038 | 5.074 ± 0.097 | 5.328 ± 0.058 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.397 ± 0.078 | 22.842 ± 0.152 | 24.869 ± 0.247 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

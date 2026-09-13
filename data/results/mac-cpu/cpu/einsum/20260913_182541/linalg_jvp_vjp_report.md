# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_182541`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_182541`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

## CPU Information

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

## Thread Environment

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 1

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_182541/cpu_ops_t1_20260913_182541.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_182541/linalg_jvp_vjp_t1_20260913_182541.md`

## Linalg JVP/VJP Benchmark Items

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

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

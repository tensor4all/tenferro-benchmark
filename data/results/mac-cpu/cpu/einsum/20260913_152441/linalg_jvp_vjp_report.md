# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_152441`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_152441`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_152441/cpu_ops_t1_20260913_152441.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_152441/linalg_jvp_vjp_t1_20260913_152441.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 77.401 ± 0.538 | 85.895 ± 1.288 | 134.321 ± 0.431 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 3.066 ± 0.085 | 3.127 ± 0.048 | 3.940 ± 0.036 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 14.166 ± 0.165 | 14.912 ± 0.118 | 21.484 ± 0.074 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 78.018 ± 1.330 | 73.734 ± 1.017 | 132.915 ± 0.248 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 3.023 ± 0.049 | 2.867 ± 0.064 | 3.919 ± 0.038 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 14.087 ± 0.106 | 13.476 ± 0.187 | 21.250 ± 0.093 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 31.655 ± 0.977 | 70.085 ± 3.526 | 86.173 ± 0.636 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.723 ± 0.052 | 1.341 ± 0.028 | 1.735 ± 0.010 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 4.376 ± 0.245 | 9.112 ± 0.252 | 11.947 ± 0.054 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 31.369 ± 0.523 | 35.086 ± 1.339 | 86.883 ± 0.263 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 0.719 ± 0.033 | 0.779 ± 0.040 | 1.744 ± 0.021 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 4.483 ± 0.172 | 5.051 ± 0.096 | 11.793 ± 0.096 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 53.897 ± 0.697 | 60.776 ± 1.281 | 134.588 ± 0.353 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.922 ± 0.105 | 1.945 ± 0.049 | 3.154 ± 0.013 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 9.347 ± 0.080 | 10.397 ± 0.186 | 18.824 ± 0.180 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 54.259 ± 1.244 | 61.842 ± 0.756 | 136.882 ± 0.356 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.912 ± 0.094 | 1.891 ± 0.087 | 3.130 ± 0.009 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 9.261 ± 0.044 | 9.913 ± 0.167 | 19.482 ± 0.069 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 6.209 ± 0.081 | 6.399 ± 0.191 | 5.185 ± 0.069 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.293 ± 0.007 | 0.320 ± 0.068 | 0.244 ± 0.016 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.250 ± 0.026 | 1.284 ± 0.108 | 1.010 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 6.131 ± 0.194 | 10.562 ± 0.157 | 4.728 ± 0.043 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.295 ± 0.013 | 0.383 ± 0.005 | 0.236 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.240 ± 0.034 | 1.967 ± 0.065 | 0.986 ± 0.016 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 121.893 ± 0.605 | 142.589 ± 1.694 | 177.237 ± 1.042 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 5.201 ± 0.097 | 5.738 ± 0.138 | 6.079 ± 0.042 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 23.931 ± 0.094 | 26.508 ± 0.086 | 30.890 ± 0.059 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 120.411 ± 0.766 | 117.949 ± 1.181 | 176.587 ± 0.472 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 5.179 ± 0.099 | 5.173 ± 0.078 | 6.022 ± 0.021 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 23.777 ± 0.073 | 23.156 ± 0.105 | 31.077 ± 0.589 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

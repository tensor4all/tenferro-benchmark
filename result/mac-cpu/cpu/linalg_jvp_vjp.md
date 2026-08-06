# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260806_150841`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260806_150841`.

- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environment

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

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260806_150841/cpu_ops_t4_20260806_150841.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260806_150841/linalg_jvp_vjp_t4_20260806_150841.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.157 ± 0.027 | 3.218 ± 0.095 | 3.645 ± 0.131 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.624 ± 0.401 | 15.274 ± 0.097 | 17.907 ± 0.191 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.115 ± 0.023 | 2.927 ± 0.103 | 3.512 ± 0.123 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.252 ± 0.092 | 13.945 ± 3.638 | 16.637 ± 0.220 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.856 ± 0.054 | 1.326 ± 0.101 | 1.130 ± 0.100 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.724 ± 0.101 | 9.497 ± 1.240 | 5.015 ± 0.160 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.825 ± 0.039 | 0.847 ± 0.017 | 1.369 ± 0.207 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.746 ± 0.082 | 4.933 ± 0.445 | 5.622 ± 0.190 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.079 ± 0.059 | 1.869 ± 0.021 | 2.111 ± 0.163 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.190 ± 0.028 | 9.185 ± 0.170 | 9.434 ± 0.157 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.067 ± 0.055 | 1.795 ± 0.045 | 2.253 ± 0.061 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.065 ± 0.065 | 8.765 ± 0.219 | 10.477 ± 0.223 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.488 ± 0.071 | 0.392 ± 0.017 | 0.443 ± 0.072 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.469 ± 0.022 | 1.294 ± 0.079 | 1.367 ± 0.081 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.446 ± 0.041 | 0.469 ± 0.032 | 0.532 ± 0.079 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.421 ± 0.028 | 2.116 ± 0.323 | 1.366 ± 0.056 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.264 ± 0.056 | 5.755 ± 0.162 | 6.084 ± 0.095 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 24.093 ± 0.140 | 26.569 ± 0.287 | 27.961 ± 0.178 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.227 ± 0.026 | 5.154 ± 0.073 | 5.751 ± 0.061 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.923 ± 0.129 | 32.055 ± 0.908 | 34.231 ± 0.337 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.148 ± 0.019 | 0.044 ± 0.004 | 0.149 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.140 ± 0.028 | 0.044 ± 0.002 | 0.147 ± 0.008 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.159 ± 0.026 | 0.051 ± 0.001 | 0.156 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.152 ± 0.035 | 0.041 ± 0.005 | 0.352 ± 0.022 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.124 ± 0.005 | 0.037 ± 0.003 | 0.363 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.151 ± 0.026 | 0.046 ± 0.002 | 0.353 ± 0.024 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.161 ± 0.022 | 0.116 ± 0.022 | 0.217 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.173 ± 0.037 | 0.135 ± 0.012 | 0.221 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.155 ± 0.017 | 0.136 ± 0.003 | 0.218 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.156 ± 0.035 | 0.090 ± 0.030 | 0.589 ± 0.030 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.156 ± 0.025 | 0.105 ± 0.005 | 0.561 ± 0.026 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.181 ± 0.008 | 0.104 ± 0.013 | 0.584 ± 0.042 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.324 ± 0.032 | 0.083 ± 0.018 | 0.212 ± 0.013 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.275 ± 0.028 | 0.068 ± 0.016 | 0.210 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.285 ± 0.031 | 0.081 ± 0.007 | 0.210 ± 0.008 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.335 ± 0.037 | 0.075 ± 0.011 | 0.541 ± 0.043 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.328 ± 0.034 | 0.079 ± 0.004 | 0.541 ± 0.031 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.319 ± 0.014 | 0.083 ± 0.003 | 0.523 ± 0.026 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.176 ± 0.028 | 0.114 ± 0.003 | 0.186 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.172 ± 0.023 | 0.111 ± 0.006 | 0.180 ± 0.016 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.178 ± 0.026 | 0.110 ± 0.004 | 0.175 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.181 ± 0.026 | 0.057 ± 0.004 | 0.398 ± 0.045 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.196 ± 0.021 | 0.054 ± 0.002 | 0.381 ± 0.039 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.167 ± 0.017 | 0.055 ± 0.005 | 0.378 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.120 ± 0.021 | 0.065 ± 0.001 | 0.159 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.116 ± 0.029 | 0.062 ± 0.002 | 0.163 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.121 ± 0.006 | 0.072 ± 0.001 | 0.158 ± 0.011 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.092 ± 0.008 | 0.051 ± 0.009 | 0.385 ± 0.021 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.101 ± 0.020 | 0.042 ± 0.003 | 0.405 ± 0.018 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.128 ± 0.047 | 0.045 ± 0.003 | 0.411 ± 0.029 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

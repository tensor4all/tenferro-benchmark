# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260729_195802`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260729_195802`.

- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260729_195802/cpu_ops_t4_20260729_195802.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260729_195802/linalg_jvp_vjp_t4_20260729_195802.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.096 ± 0.099 | 3.347 ± 0.133 | 3.444 ± 0.052 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.882 ± 0.812 | 14.183 ± 0.053 | 14.234 ± 0.269 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.133 ± 0.185 | 2.811 ± 0.049 | 3.379 ± 0.078 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.656 ± 0.035 | 12.745 ± 0.091 | 14.299 ± 0.246 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.087 ± 0.063 | 1.435 ± 0.098 | 0.992 ± 0.059 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 5.153 ± 0.565 | 7.708 ± 0.081 | 3.804 ± 0.545 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.936 ± 0.064 | 0.896 ± 0.047 | 1.226 ± 0.190 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.706 ± 0.215 | 4.331 ± 0.083 | 4.011 ± 0.291 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.041 ± 0.043 | 1.889 ± 0.043 | 2.054 ± 0.119 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.657 ± 0.220 | 8.542 ± 0.023 | 7.442 ± 0.546 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.037 ± 0.020 | 1.795 ± 0.069 | 2.152 ± 0.066 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 11.607 ± 3.196 | 8.253 ± 0.030 | 8.438 ± 0.406 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.410 ± 0.021 | 0.423 ± 0.048 | 0.405 ± 0.029 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.297 ± 0.012 | 1.187 ± 0.091 | 1.308 ± 0.084 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.409 ± 0.014 | 0.460 ± 0.023 | 0.480 ± 0.056 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.304 ± 0.045 | 1.826 ± 0.057 | 1.394 ± 0.049 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.184 ± 0.042 | 5.589 ± 0.068 | 5.911 ± 0.077 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.422 ± 0.471 | 25.395 ± 0.918 | 24.633 ± 0.153 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.097 ± 0.026 | 4.986 ± 0.032 | 5.612 ± 0.102 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.354 ± 0.259 | 22.633 ± 0.025 | 23.776 ± 0.025 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.110 ± 0.007 | 0.035 ± 0.001 | 0.136 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.112 ± 0.018 | 0.037 ± 0.001 | 0.140 ± 0.071 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.115 ± 0.014 | 0.043 ± 0.001 | 0.134 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.147 ± 0.048 | 0.033 ± 0.003 | 0.311 ± 0.015 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.111 ± 0.007 | 0.032 ± 0.001 | 0.342 ± 0.081 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.113 ± 0.007 | 0.038 ± 0.006 | 0.325 ± 0.026 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.121 ± 0.005 | 0.147 ± 0.026 | 0.185 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.106 ± 0.008 | 0.129 ± 0.010 | 0.186 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.109 ± 0.018 | 0.137 ± 0.011 | 0.187 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.117 ± 0.005 | 0.105 ± 0.017 | 0.514 ± 0.018 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.111 ± 0.008 | 0.116 ± 0.011 | 0.519 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.006 | 0.101 ± 0.005 | 0.538 ± 0.059 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.202 ± 0.019 | 0.084 ± 0.018 | 0.184 ± 0.018 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.205 ± 0.015 | 0.079 ± 0.005 | 0.182 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.198 ± 0.005 | 0.079 ± 0.003 | 0.190 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.201 ± 0.010 | 0.079 ± 0.010 | 0.476 ± 0.028 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.200 ± 0.025 | 0.084 ± 0.006 | 0.473 ± 0.026 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.202 ± 0.009 | 0.078 ± 0.005 | 0.482 ± 0.028 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.140 ± 0.013 | 0.095 ± 0.006 | 0.147 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.126 ± 0.007 | 0.098 ± 0.014 | 0.157 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.127 ± 0.008 | 0.095 ± 0.003 | 0.145 ± 0.004 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.165 ± 0.019 | 0.049 ± 0.007 | 0.335 ± 0.020 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.150 ± 0.008 | 0.045 ± 0.005 | 0.365 ± 0.100 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.143 ± 0.008 | 0.047 ± 0.003 | 0.345 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.062 ± 0.005 | 0.051 ± 0.005 | 0.125 ± 0.007 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.067 ± 0.005 | 0.053 ± 0.004 | 0.130 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.074 ± 0.005 | 0.062 ± 0.002 | 0.131 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.078 ± 0.014 | 0.040 ± 0.012 | 0.333 ± 0.013 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.066 ± 0.001 | 0.035 ± 0.002 | 0.363 ± 0.062 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.076 ± 0.008 | 0.040 ± 0.002 | 0.357 ± 0.042 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

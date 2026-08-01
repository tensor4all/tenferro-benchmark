# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260801_221400`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260801_221400`.

- tenferro-rs commit: `0ee2d0dc2f8d21ff62ea682f90f34e4319108ace`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260801_221400/cpu_ops_t4_20260801_221400.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260801_221400/linalg_jvp_vjp_t4_20260801_221400.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.153 ± 0.038 | 3.098 ± 0.012 | 11.009 ± 0.974 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.304 ± 0.111 | 14.620 ± 0.044 | 49.045 ± 4.577 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.119 ± 0.032 | 2.848 ± 0.031 | 17.440 ± 7.800 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.220 ± 0.023 | 13.056 ± 0.170 | 38.447 ± 0.768 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.836 ± 0.059 | 1.295 ± 0.013 | 5.133 ± 1.114 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.828 ± 0.300 | 7.979 ± 0.042 | 13.408 ± 0.568 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.773 ± 0.011 | 0.831 ± 0.013 | 4.217 ± 0.154 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.541 ± 0.053 | 4.517 ± 0.017 | 14.684 ± 1.175 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.982 ± 0.035 | 1.848 ± 0.021 | 88.564 ± 46.677 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.968 ± 0.194 | 8.738 ± 0.063 | 13.000 ± 0.883 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.966 ± 0.042 | 1.771 ± 0.035 | 8.878 ± 1.713 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.727 ± 0.028 | 8.363 ± 0.061 | 13.830 ± 0.193 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.458 ± 0.068 | 0.377 ± 0.007 | 1.023 ± 0.103 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.378 ± 0.007 | 1.286 ± 0.044 | 3.197 ± 0.185 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.449 ± 0.078 | 0.447 ± 0.006 | 1.299 ± 0.067 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.389 ± 0.017 | 1.995 ± 0.024 | 2.573 ± 0.132 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.231 ± 0.034 | 5.667 ± 0.041 | 162.323 ± 153.446 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 24.032 ± 0.032 | 26.169 ± 0.066 | 44.218 ± 2.311 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.218 ± 0.006 | 5.138 ± 0.024 | 302.928 ± 147.727 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 31.223 ± 7.563 | 30.670 ± 0.069 | 44.899 ± 3.259 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.108 ± 0.004 | 0.045 ± 0.003 | 0.151 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.102 ± 0.007 | 0.044 ± 0.001 | 0.147 ± 0.010 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.112 ± 0.001 | 0.051 ± 0.001 | 0.499 ± 0.410 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.106 ± 0.002 | 0.040 ± 0.005 | 0.363 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.106 ± 0.005 | 0.038 ± 0.002 | 0.362 ± 0.022 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.111 ± 0.002 | 0.045 ± 0.002 | 2.277 ± 1.698 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.105 ± 0.003 | 0.112 ± 0.010 | 0.217 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.100 ± 0.004 | 0.107 ± 0.007 | 0.216 ± 0.007 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.098 ± 0.013 | 0.138 ± 0.006 | 3.080 ± 3.954 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.120 ± 0.004 | 0.083 ± 0.012 | 0.589 ± 0.038 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.119 ± 0.003 | 0.095 ± 0.018 | 0.586 ± 0.026 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.130 ± 0.005 | 0.112 ± 0.016 | 1.037 ± 0.291 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.224 ± 0.018 | 0.080 ± 0.017 | 0.218 ± 0.012 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.219 ± 0.014 | 0.069 ± 0.006 | 0.221 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.217 ± 0.006 | 0.079 ± 0.002 | 70.858 ± 135.975 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.240 ± 0.006 | 0.077 ± 0.018 | 0.555 ± 0.041 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.238 ± 0.009 | 0.069 ± 0.005 | 0.534 ± 0.026 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.246 ± 0.004 | 0.079 ± 0.011 | 6.297 ± 3.003 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.103 ± 0.001 | 0.107 ± 0.005 | 0.178 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.103 ± 0.007 | 0.106 ± 0.003 | 0.178 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.102 ± 0.013 | 0.105 ± 0.003 | 0.316 ± 0.062 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.112 ± 0.001 | 0.053 ± 0.002 | 0.380 ± 0.034 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.115 ± 0.006 | 0.053 ± 0.002 | 0.389 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.113 ± 0.009 | 0.054 ± 0.005 | 1.253 ± 1.043 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.104 ± 0.006 | 0.064 ± 0.005 | 0.161 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.102 ± 0.001 | 0.063 ± 0.001 | 0.159 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.125 ± 0.008 | 0.070 ± 0.001 | 44.674 ± 47.250 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.096 ± 0.011 | 0.048 ± 0.008 | 0.389 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.101 ± 0.022 | 0.042 ± 0.003 | 0.399 ± 0.027 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.112 ± 0.007 | 0.045 ± 0.002 | 2.797 ± 0.761 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260723_121605`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260723_121605`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260723_121605/cpu_ops_t4_20260723_121605.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260723_121605/linalg_jvp_vjp_t4_20260723_121605.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.163 ± 0.007 | 3.204 ± 0.027 | 3.785 ± 0.089 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.669 ± 0.172 | 15.374 ± 0.224 | 18.053 ± 0.968 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.141 ± 0.074 | 2.905 ± 0.080 | 3.599 ± 0.069 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 15.348 ± 3.016 | 13.739 ± 0.306 | 16.932 ± 0.196 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.863 ± 0.038 | 1.343 ± 0.126 | 1.151 ± 0.082 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 5.591 ± 0.128 | 8.610 ± 0.484 | 5.475 ± 0.686 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.872 ± 0.025 | 0.884 ± 0.071 | 1.406 ± 0.185 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 5.119 ± 0.154 | 4.967 ± 0.185 | 5.754 ± 0.328 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.896 ± 0.022 | 1.870 ± 0.070 | 2.221 ± 0.072 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.142 ± 0.189 | 9.381 ± 0.522 | 11.073 ± 0.341 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.993 ± 0.053 | 1.797 ± 0.077 | 2.320 ± 0.135 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.294 ± 0.179 | 8.677 ± 0.127 | 11.262 ± 0.632 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.247 ± 0.004 | 0.414 ± 0.061 | 0.425 ± 0.053 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.361 ± 0.013 | 1.366 ± 0.127 | 1.376 ± 0.100 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.254 ± 0.014 | 0.474 ± 0.013 | 0.672 ± 0.102 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.178 ± 0.054 | 2.080 ± 0.102 | 1.421 ± 0.065 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.207 ± 0.050 | 5.731 ± 0.117 | 6.151 ± 0.130 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 32.166 ± 0.151 | 26.624 ± 0.451 | 28.284 ± 0.860 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.188 ± 0.007 | 5.129 ± 0.057 | 5.824 ± 0.179 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 31.612 ± 0.283 | 31.523 ± 0.175 | 34.434 ± 0.543 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.051 ± 0.011 | 0.043 ± 0.003 | 0.161 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.059 ± 0.006 | 0.045 ± 0.003 | 0.156 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.056 ± 0.025 | 0.050 ± 0.002 | 0.157 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.058 ± 0.024 | 0.041 ± 0.008 | 0.374 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.083 ± 0.022 | 0.040 ± 0.007 | 0.351 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.091 ± 0.018 | 0.045 ± 0.002 | 0.375 ± 0.025 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.039 ± 0.012 | 0.113 ± 0.005 | 0.227 ± 0.017 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.039 ± 0.008 | 0.132 ± 0.010 | 0.208 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.039 ± 0.023 | 0.130 ± 0.004 | 0.226 ± 0.013 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.038 ± 0.003 | 0.085 ± 0.013 | 0.592 ± 0.046 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.035 ± 0.007 | 0.111 ± 0.017 | 0.573 ± 0.029 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.043 ± 0.003 | 0.105 ± 0.009 | 0.574 ± 0.023 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.113 ± 0.010 | 0.080 ± 0.036 | 0.211 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.107 ± 0.017 | 0.079 ± 0.011 | 0.232 ± 0.111 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.107 ± 0.012 | 0.079 ± 0.014 | 0.216 ± 0.011 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.164 ± 0.012 | 0.086 ± 0.023 | 0.540 ± 0.038 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.227 ± 0.034 | 0.082 ± 0.013 | 0.525 ± 0.038 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.200 ± 0.018 | 0.090 ± 0.015 | 0.527 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.031 ± 0.012 | 0.111 ± 0.007 | 0.180 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.022 ± 0.015 | 0.108 ± 0.003 | 0.186 ± 0.019 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.034 ± 0.006 | 0.108 ± 0.003 | 0.176 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.040 ± 0.014 | 0.055 ± 0.004 | 0.383 ± 0.029 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.035 ± 0.015 | 0.055 ± 0.010 | 0.384 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.039 ± 0.005 | 0.060 ± 0.006 | 0.391 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.031 ± 0.007 | 0.064 ± 0.004 | 0.152 ± 0.010 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.022 ± 0.009 | 0.061 ± 0.003 | 0.159 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.035 ± 0.033 | 0.071 ± 0.003 | 0.157 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.026 ± 0.013 | 0.050 ± 0.009 | 0.406 ± 0.021 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.026 ± 0.026 | 0.041 ± 0.003 | 0.410 ± 0.055 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.033 ± 0.003 | 0.046 ± 0.002 | 0.415 ± 0.015 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260822_090304`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260822_090304`.

- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260822_090304/cpu_ops_t4_20260822_090304.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260822_090304/linalg_jvp_vjp_t4_20260822_090304.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.162 ± 0.036 | 3.149 ± 0.091 | 4.088 ± 0.190 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.466 ± 0.242 | 15.122 ± 0.456 | 17.799 ± 0.496 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.161 ± 0.102 | 2.927 ± 0.083 | 7.303 ± 0.498 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.745 ± 0.149 | 13.565 ± 0.259 | 17.001 ± 0.413 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.879 ± 0.013 | 1.312 ± 0.146 | 2.170 ± 0.705 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 5.046 ± 0.191 | 8.455 ± 0.437 | 5.131 ± 0.676 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.835 ± 0.015 | 0.848 ± 0.027 | 1.415 ± 0.284 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 5.067 ± 0.235 | 4.749 ± 0.304 | 6.342 ± 0.390 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.062 ± 0.051 | 1.868 ± 0.047 | 2.263 ± 0.106 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.074 ± 0.397 | 9.057 ± 0.183 | 9.422 ± 0.200 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.080 ± 0.046 | 1.808 ± 0.023 | 2.463 ± 0.220 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.043 ± 0.059 | 8.531 ± 0.108 | 10.542 ± 0.418 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.421 ± 0.031 | 0.381 ± 0.009 | 0.452 ± 0.084 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.467 ± 0.059 | 1.307 ± 0.045 | 1.362 ± 0.081 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.440 ± 0.067 | 0.453 ± 0.029 | 0.588 ± 0.175 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.487 ± 0.057 | 2.080 ± 0.087 | 1.405 ± 0.063 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.267 ± 0.026 | 5.701 ± 0.096 | 6.094 ± 0.063 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 24.303 ± 0.146 | 26.499 ± 0.151 | 28.140 ± 0.618 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.242 ± 0.035 | 5.134 ± 0.040 | 5.855 ± 0.193 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 24.612 ± 7.793 | 31.200 ± 0.399 | 33.712 ± 0.397 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.184 ± 0.026 | 0.044 ± 0.002 | 0.149 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.177 ± 0.014 | 0.044 ± 0.002 | 0.153 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.176 ± 0.020 | 0.051 ± 0.001 | 0.156 ± 0.011 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.152 ± 0.013 | 0.040 ± 0.004 | 0.346 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.159 ± 0.015 | 0.039 ± 0.001 | 0.365 ± 0.019 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.181 ± 0.015 | 0.045 ± 0.001 | 0.363 ± 0.028 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.165 ± 0.018 | 0.116 ± 0.033 | 0.208 ± 0.007 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.160 ± 0.013 | 0.132 ± 0.008 | 0.218 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.180 ± 0.022 | 0.137 ± 0.011 | 0.220 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.172 ± 0.034 | 0.088 ± 0.027 | 0.601 ± 0.047 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.176 ± 0.012 | 0.107 ± 0.029 | 0.618 ± 0.081 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.198 ± 0.041 | 0.110 ± 0.007 | 0.630 ± 0.021 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.303 ± 0.015 | 0.078 ± 0.028 | 0.213 ± 0.012 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.323 ± 0.023 | 0.078 ± 0.007 | 0.216 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.309 ± 0.021 | 0.081 ± 0.003 | 0.219 ± 0.020 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.398 ± 0.096 | 0.075 ± 0.027 | 0.537 ± 0.056 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.355 ± 0.070 | 0.095 ± 0.033 | 0.530 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.314 ± 0.022 | 0.087 ± 0.011 | 0.533 ± 0.027 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.167 ± 0.009 | 0.110 ± 0.008 | 0.176 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.157 ± 0.036 | 0.109 ± 0.003 | 0.187 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.156 ± 0.021 | 0.112 ± 0.003 | 0.183 ± 0.015 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.173 ± 0.031 | 0.054 ± 0.003 | 0.392 ± 0.029 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.157 ± 0.021 | 0.054 ± 0.003 | 0.396 ± 0.029 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.169 ± 0.005 | 0.056 ± 0.007 | 0.407 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.110 ± 0.007 | 0.064 ± 0.006 | 0.165 ± 0.015 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.108 ± 0.018 | 0.060 ± 0.002 | 0.161 ± 0.018 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.122 ± 0.006 | 0.070 ± 0.002 | 0.159 ± 0.006 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.100 ± 0.004 | 0.050 ± 0.008 | 0.400 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.107 ± 0.012 | 0.041 ± 0.003 | 0.413 ± 0.020 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.104 ± 0.016 | 0.046 ± 0.002 | 0.419 ± 0.019 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260716_171807`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260716_171807`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260716_171807/cpu_ops_t4_20260716_171807.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260716_171807/linalg_jvp_vjp_t4_20260716_171807.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.409 ± 0.301 | 3.244 ± 0.052 | 3.822 ± 0.455 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 15.022 ± 0.113 | 15.449 ± 0.184 | 17.894 ± 0.370 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.194 ± 0.276 | 2.957 ± 0.078 | 3.673 ± 0.086 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.898 ± 0.052 | 13.948 ± 0.213 | 16.951 ± 0.189 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.872 ± 0.035 | 1.360 ± 0.048 | 1.136 ± 0.039 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 9.339 ± 7.509 | 8.630 ± 0.934 | 5.317 ± 0.369 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.921 ± 0.108 | 0.853 ± 0.026 | 1.340 ± 0.580 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 5.198 ± 0.618 | 4.998 ± 0.301 | 5.846 ± 0.686 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.945 ± 0.004 | 1.893 ± 0.087 | 2.329 ± 0.305 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.403 ± 0.121 | 9.211 ± 0.204 | 10.523 ± 1.451 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.039 ± 0.048 | 1.792 ± 0.054 | 4.552 ± 1.869 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.273 ± 0.063 | 8.773 ± 0.356 | 10.837 ± 0.608 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.262 ± 0.008 | 0.400 ± 0.050 | 0.468 ± 0.062 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.426 ± 0.221 | 1.341 ± 0.149 | 1.378 ± 0.120 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.301 ± 0.044 | 0.459 ± 0.018 | 0.501 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.385 ± 0.138 | 2.123 ± 0.180 | 1.399 ± 0.188 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.259 ± 0.039 | 5.792 ± 0.063 | 6.251 ± 0.104 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 25.045 ± 0.521 | 26.915 ± 0.273 | 28.483 ± 0.397 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.205 ± 0.037 | 5.221 ± 0.081 | 5.965 ± 0.169 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 32.078 ± 0.235 | 31.634 ± 0.575 | 34.934 ± 0.923 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.058 ± 0.013 | 0.043 ± 0.003 | 0.153 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.054 ± 0.018 | 0.045 ± 0.003 | 0.172 ± 0.053 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.074 ± 0.038 | 0.051 ± 0.002 | 0.158 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.092 ± 0.032 | 0.040 ± 0.008 | 0.353 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.092 ± 0.017 | 0.040 ± 0.009 | 0.353 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.065 ± 0.049 | 0.045 ± 0.002 | 0.394 ± 0.086 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.044 ± 0.018 | 0.110 ± 0.004 | 0.220 ± 0.013 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.035 ± 0.006 | 0.132 ± 0.006 | 0.215 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.086 ± 0.048 | 0.136 ± 0.006 | 0.212 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.048 ± 0.034 | 0.087 ± 0.015 | 0.571 ± 0.011 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.041 ± 0.010 | 0.108 ± 0.011 | 0.584 ± 0.036 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.103 ± 0.016 | 0.108 ± 0.005 | 0.629 ± 0.100 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.113 ± 0.021 | 0.086 ± 0.013 | 0.214 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.098 ± 0.039 | 0.070 ± 0.008 | 0.226 ± 0.026 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.153 ± 0.057 | 0.082 ± 0.025 | 0.216 ± 0.010 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.203 ± 0.007 | 0.076 ± 0.026 | 0.521 ± 0.041 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.202 ± 0.004 | 0.082 ± 0.044 | 0.534 ± 0.023 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.183 ± 0.008 | 0.088 ± 0.020 | 0.667 ± 0.246 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.025 ± 0.011 | 0.123 ± 0.045 | 0.177 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.020 ± 0.005 | 0.110 ± 0.005 | 0.174 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.044 ± 0.046 | 0.111 ± 0.004 | 0.193 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.073 ± 0.005 | 0.060 ± 0.009 | 0.376 ± 0.039 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.066 ± 0.024 | 0.055 ± 0.003 | 0.499 ± 0.171 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.199 ± 0.168 | 0.058 ± 0.003 | 0.429 ± 0.144 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.023 ± 0.002 | 0.063 ± 0.005 | 0.165 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.026 ± 0.010 | 0.064 ± 0.004 | 0.162 ± 0.029 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.040 ± 0.013 | 0.071 ± 0.002 | 0.160 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.025 ± 0.008 | 0.051 ± 0.011 | 0.381 ± 0.025 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.025 ± 0.005 | 0.043 ± 0.002 | 0.401 ± 0.029 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.051 ± 0.015 | 0.046 ± 0.003 | 0.401 ± 0.026 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260630_071006`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260630_071006`.

- tenferro-rs commit: `3b4136ca3a3d53cbbdd5096954a470d9407ad25e`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260630_071006/cpu_ops_t4_20260630_071006.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260630_071006/linalg_jvp_vjp_t4_20260630_071006.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 9.800 ± 0.628 | 3.272 ± 0.214 | 3.604 ± 0.233 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 64.006 ± 0.664 | 14.406 ± 0.087 | 14.345 ± 0.232 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 9.811 ± 0.058 | 2.905 ± 0.104 | 3.469 ± 0.113 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 65.847 ± 0.191 | 13.007 ± 0.160 | 14.127 ± 0.151 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.250 ± 0.044 | 1.423 ± 0.064 | 1.035 ± 0.102 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 6.097 ± 0.059 | 8.239 ± 0.371 | 3.550 ± 0.097 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.056 ± 0.147 | 0.923 ± 0.083 | 1.182 ± 0.057 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 10.382 ± 0.062 | 4.515 ± 0.145 | 3.919 ± 0.212 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.891 ± 0.050 | 1.913 ± 0.097 | 2.053 ± 0.120 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.086 ± 0.712 | 8.725 ± 0.150 | 7.645 ± 0.233 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.700 ± 0.303 | 1.824 ± 0.071 | 2.218 ± 0.068 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 18.121 ± 7.595 | 8.314 ± 0.098 | 8.289 ± 0.137 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.254 ± 0.016 | 0.430 ± 0.113 | 0.417 ± 0.109 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.120 ± 0.008 | 1.180 ± 0.085 | 1.321 ± 0.137 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.312 ± 0.008 | 0.488 ± 0.050 | 0.531 ± 0.152 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.236 ± 0.014 | 1.832 ± 0.048 | 1.238 ± 0.103 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 60.062 ± 1.138 | 5.630 ± 0.061 | 5.877 ± 0.025 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 595.751 ± 14.842 | 25.700 ± 0.346 | 25.707 ± 0.619 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 58.916 ± 0.996 | 5.039 ± 0.084 | 6.094 ± 1.177 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 591.349 ± 4.332 | 22.702 ± 0.041 | 24.110 ± 0.144 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.027 ± 0.002 | 0.038 ± 0.004 | 0.130 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.029 ± 0.002 | 0.036 ± 0.004 | 0.127 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.029 ± 0.001 | 0.045 ± 0.001 | 0.129 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.051 ± 0.003 | 0.034 ± 0.011 | 0.331 ± 0.092 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.052 ± 0.003 | 0.042 ± 0.031 | 0.317 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.064 ± 0.005 | 0.040 ± 0.004 | 0.360 ± 0.182 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.051 ± 0.005 | 0.135 ± 0.002 | 0.191 ± 0.030 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.049 ± 0.005 | 0.136 ± 0.005 | 0.187 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.050 ± 0.006 | 0.145 ± 0.010 | 0.189 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.072 ± 0.007 | 0.106 ± 0.010 | 0.535 ± 0.083 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.069 ± 0.007 | 0.104 ± 0.006 | 0.534 ± 0.034 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.078 ± 0.005 | 0.104 ± 0.010 | 0.684 ± 0.239 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.077 ± 0.003 | 0.082 ± 0.008 | 0.192 ± 0.076 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.073 ± 0.004 | 0.093 ± 0.023 | 0.187 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.076 ± 0.001 | 0.082 ± 0.011 | 0.185 ± 0.008 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.132 ± 0.003 | 0.082 ± 0.021 | 0.473 ± 0.029 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.130 ± 0.011 | 0.081 ± 0.012 | 0.481 ± 0.085 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.135 ± 0.014 | 0.079 ± 0.009 | 0.482 ± 0.029 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.023 ± 0.004 | 0.093 ± 0.002 | 0.155 ± 0.020 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.025 ± 0.008 | 0.095 ± 0.004 | 0.156 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.024 ± 0.003 | 0.091 ± 0.004 | 0.169 ± 0.087 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.046 ± 0.006 | 0.046 ± 0.003 | 0.371 ± 0.180 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.045 ± 0.006 | 0.047 ± 0.011 | 0.376 ± 0.207 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.047 ± 0.003 | 0.048 ± 0.005 | 0.357 ± 0.023 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.035 ± 0.023 | 0.053 ± 0.004 | 0.134 ± 0.014 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.019 ± 0.001 | 0.053 ± 0.002 | 0.134 ± 0.005 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.020 ± 0.001 | 0.066 ± 0.025 | 0.136 ± 0.035 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.032 ± 0.003 | 0.045 ± 0.022 | 0.341 ± 0.016 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.032 ± 0.002 | 0.039 ± 0.008 | 0.346 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.038 ± 0.001 | 0.040 ± 0.002 | 0.356 ± 0.020 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

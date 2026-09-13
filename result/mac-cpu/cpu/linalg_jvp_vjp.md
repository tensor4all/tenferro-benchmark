# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_124035`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_124035`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_124035/cpu_ops_t4_20260913_124035.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_124035/linalg_jvp_vjp_t4_20260913_124035.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 79.280 ± 2.002 | 84.966 ± 3.913 | 75.984 ± 0.875 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.144 ± 0.065 | 3.200 ± 0.103 | 3.550 ± 0.118 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.649 ± 0.086 | 15.826 ± 0.941 | 14.546 ± 0.091 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 76.930 ± 3.692 | 72.221 ± 2.275 | 74.733 ± 0.309 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.166 ± 0.054 | 2.897 ± 0.098 | 3.610 ± 0.109 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.624 ± 0.094 | 13.733 ± 1.050 | 14.557 ± 0.297 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 32.616 ± 2.297 | 61.391 ± 4.324 | 21.579 ± 0.473 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.925 ± 0.048 | 1.411 ± 0.115 | 1.063 ± 0.093 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.474 ± 0.163 | 8.170 ± 0.104 | 3.981 ± 0.145 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 32.081 ± 1.996 | 31.195 ± 1.988 | 23.571 ± 0.811 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.888 ± 0.129 | 0.894 ± 0.037 | 1.291 ± 0.095 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.242 ± 0.055 | 4.526 ± 0.068 | 4.220 ± 0.174 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 51.691 ± 3.883 | 54.421 ± 1.679 | 39.112 ± 1.415 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.142 ± 0.089 | 1.962 ± 0.055 | 2.138 ± 0.117 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.495 ± 0.080 | 8.671 ± 0.049 | 7.692 ± 0.188 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 59.124 ± 3.427 | 57.644 ± 5.196 | 44.912 ± 0.885 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.161 ± 0.053 | 1.836 ± 0.123 | 2.269 ± 0.196 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.353 ± 0.139 | 8.349 ± 0.120 | 8.811 ± 0.236 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.921 ± 0.125 | 6.496 ± 0.539 | 6.133 ± 0.302 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.458 ± 0.049 | 0.375 ± 0.021 | 0.436 ± 0.073 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.303 ± 0.017 | 1.219 ± 0.069 | 1.431 ± 0.125 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.826 ± 0.197 | 10.538 ± 0.375 | 6.183 ± 0.240 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.454 ± 0.046 | 0.476 ± 0.037 | 0.464 ± 0.106 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.267 ± 0.012 | 1.856 ± 0.081 | 1.480 ± 0.207 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 122.109 ± 3.053 | 139.852 ± 6.233 | 126.327 ± 0.903 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.249 ± 0.057 | 5.728 ± 0.063 | 6.100 ± 0.200 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.984 ± 0.462 | 25.698 ± 0.155 | 25.496 ± 0.212 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 121.107 ± 2.146 | 117.767 ± 9.731 | 120.394 ± 1.236 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.215 ± 0.047 | 5.087 ± 0.077 | 5.876 ± 0.105 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 23.120 ± 0.048 | 22.887 ± 0.104 | 24.450 ± 0.318 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.192 ± 0.007 | 0.051 ± 0.003 | 0.129 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.188 ± 0.027 | 0.036 ± 0.003 | 0.130 ± 0.016 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.222 ± 0.014 | 0.075 ± 0.002 | 0.181 ± 0.022 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.189 ± 0.019 | 0.034 ± 0.001 | 0.129 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.183 ± 0.009 | 0.043 ± 0.002 | 0.130 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.181 ± 0.005 | 0.046 ± 0.004 | 0.326 ± 0.016 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.173 ± 0.011 | 0.033 ± 0.005 | 0.307 ± 0.016 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.216 ± 0.010 | 0.075 ± 0.004 | 0.331 ± 0.033 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.174 ± 0.017 | 0.030 ± 0.001 | 0.315 ± 0.013 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.178 ± 0.010 | 0.039 ± 0.001 | 0.319 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.194 ± 0.013 | 0.139 ± 0.008 | 0.187 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.187 ± 0.042 | 0.134 ± 0.010 | 0.181 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.204 ± 0.010 | 0.150 ± 0.006 | 0.203 ± 0.099 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.193 ± 0.024 | 0.133 ± 0.007 | 0.188 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.184 ± 0.010 | 0.139 ± 0.019 | 0.191 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.191 ± 0.007 | 0.102 ± 0.006 | 0.512 ± 0.024 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.192 ± 0.014 | 0.103 ± 0.014 | 0.513 ± 0.017 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.211 ± 0.005 | 0.114 ± 0.012 | 0.570 ± 0.184 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.195 ± 0.014 | 0.106 ± 0.025 | 0.499 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.195 ± 0.013 | 0.103 ± 0.004 | 0.517 ± 0.043 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.350 ± 0.022 | 0.083 ± 0.005 | 0.186 ± 0.004 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.351 ± 0.039 | 0.103 ± 0.025 | 0.184 ± 0.012 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.404 ± 0.030 | 0.089 ± 0.009 | 0.193 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.339 ± 0.016 | 0.073 ± 0.011 | 0.184 ± 0.029 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.350 ± 0.017 | 0.081 ± 0.006 | 0.192 ± 0.014 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.368 ± 0.010 | 0.085 ± 0.009 | 0.471 ± 0.044 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.378 ± 0.052 | 0.080 ± 0.015 | 0.466 ± 0.017 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.401 ± 0.024 | 0.093 ± 0.021 | 0.525 ± 0.152 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.365 ± 0.015 | 0.075 ± 0.009 | 0.458 ± 0.021 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.364 ± 0.013 | 0.073 ± 0.013 | 0.482 ± 0.034 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.176 ± 0.007 | 0.095 ± 0.005 | 0.164 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.181 ± 0.019 | 0.093 ± 0.004 | 0.154 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.189 ± 0.010 | 0.105 ± 0.006 | 0.162 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.180 ± 0.011 | 0.095 ± 0.006 | 0.150 ± 0.011 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.180 ± 0.013 | 0.094 ± 0.003 | 0.151 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.173 ± 0.011 | 0.050 ± 0.003 | 0.353 ± 0.032 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.179 ± 0.011 | 0.047 ± 0.003 | 0.347 ± 0.011 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.182 ± 0.004 | 0.065 ± 0.004 | 0.352 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.177 ± 0.011 | 0.046 ± 0.002 | 0.335 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.176 ± 0.005 | 0.047 ± 0.003 | 0.336 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.143 ± 0.007 | 0.077 ± 0.002 | 0.146 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.122 ± 0.011 | 0.051 ± 0.003 | 0.136 ± 0.018 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.188 ± 0.007 | 0.125 ± 0.005 | 0.221 ± 0.045 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.126 ± 0.008 | 0.051 ± 0.002 | 0.136 ± 0.009 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.130 ± 0.008 | 0.061 ± 0.002 | 0.137 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.129 ± 0.005 | 0.058 ± 0.003 | 0.339 ± 0.022 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.108 ± 0.005 | 0.034 ± 0.007 | 0.341 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.172 ± 0.016 | 0.104 ± 0.002 | 0.361 ± 0.022 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.114 ± 0.016 | 0.036 ± 0.003 | 0.333 ± 0.014 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.111 ± 0.006 | 0.038 ± 0.002 | 0.350 ± 0.015 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

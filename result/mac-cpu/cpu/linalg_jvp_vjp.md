# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260614_142926`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260614_142926`.

- tenferro-rs commit: `db1990549801351308b631aef1bbca292d11a457`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260614_142926/cpu_ops_t4_20260614_142926.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260614_142926/linalg_jvp_vjp_t4_20260614_142926.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 10.928 ± 0.313 | 3.316 ± 2.032 | 3.778 ± 0.230 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 69.533 ± 1.380 | 15.116 ± 0.281 | 17.923 ± 0.593 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 11.128 ± 0.310 | 3.410 ± 0.588 | 3.570 ± 0.194 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 72.390 ± 2.796 | 13.694 ± 0.160 | 17.156 ± 0.189 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.163 ± 0.076 | 1.885 ± 0.864 | 1.135 ± 0.115 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 6.191 ± 0.512 | 8.421 ± 0.449 | 5.632 ± 1.038 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.923 ± 0.141 | 0.805 ± 0.070 | 1.376 ± 0.212 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 10.868 ± 2.034 | 4.746 ± 0.447 | 5.766 ± 0.452 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.870 ± 0.072 | 1.874 ± 0.022 | 2.325 ± 0.070 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.016 ± 0.402 | 9.387 ± 0.752 | 9.923 ± 0.271 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.543 ± 0.112 | 1.805 ± 0.087 | 2.289 ± 0.168 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 12.691 ± 5.814 | 8.740 ± 0.253 | 10.767 ± 0.365 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.286 ± 0.031 | 0.440 ± 0.057 | 0.448 ± 0.052 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.381 ± 0.037 | 1.317 ± 0.012 | 1.405 ± 0.208 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.317 ± 0.042 | 0.468 ± 0.013 | 0.565 ± 0.070 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.444 ± 0.055 | 2.037 ± 0.095 | 1.415 ± 0.268 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 63.845 ± 0.613 | 5.777 ± 0.061 | 6.275 ± 0.199 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 613.462 ± 1.333 | 41.603 ± 29.579 | 28.888 ± 0.553 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 63.992 ± 0.916 | 5.196 ± 0.108 | 5.952 ± 0.066 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 614.640 ± 2.705 | 31.808 ± 0.382 | 34.753 ± 0.270 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.036 ± 0.015 | 0.046 ± 0.004 | 0.162 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.025 ± 0.032 | 0.046 ± 0.003 | 0.170 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.029 ± 0.009 | 0.053 ± 0.002 | 0.164 ± 0.006 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.043 ± 0.004 | 0.041 ± 0.001 | 0.380 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.042 ± 0.005 | 0.040 ± 0.002 | 0.409 ± 0.098 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.050 ± 0.004 | 0.049 ± 0.019 | 0.379 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.043 ± 0.010 | 0.116 ± 0.009 | 0.236 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.046 ± 0.017 | 0.135 ± 0.009 | 0.230 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.042 ± 0.008 | 0.139 ± 0.015 | 0.237 ± 0.007 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.056 ± 0.012 | 0.092 ± 0.008 | 0.606 ± 0.017 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.056 ± 0.005 | 0.105 ± 0.000 | 0.617 ± 0.285 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.061 ± 0.007 | 0.115 ± 0.009 | 0.628 ± 0.019 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.072 ± 0.039 | 0.079 ± 0.029 | 0.252 ± 0.061 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.061 ± 0.013 | 0.086 ± 0.020 | 0.239 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.066 ± 0.025 | 0.088 ± 0.011 | 0.248 ± 0.051 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.114 ± 0.014 | 0.082 ± 0.021 | 0.594 ± 0.096 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.123 ± 0.015 | 0.087 ± 0.008 | 0.579 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.122 ± 0.015 | 0.086 ± 0.007 | 0.574 ± 0.016 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.022 ± 0.010 | 0.121 ± 0.005 | 0.198 ± 0.007 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.034 ± 0.001 | 0.118 ± 0.004 | 0.193 ± 0.019 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.032 ± 0.036 | 0.118 ± 0.003 | 0.190 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.037 ± 0.009 | 0.056 ± 0.003 | 0.407 ± 0.036 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.038 ± 0.003 | 0.056 ± 0.003 | 0.416 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.038 ± 0.010 | 0.058 ± 0.003 | 0.422 ± 0.037 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.013 | 0.066 ± 0.006 | 0.173 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.012 | 0.066 ± 0.003 | 0.170 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.023 ± 0.001 | 0.072 ± 0.002 | 0.170 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.028 ± 0.002 | 0.046 ± 0.007 | 0.436 ± 0.115 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.029 ± 0.006 | 0.045 ± 0.002 | 0.457 ± 0.244 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.024 | 0.048 ± 0.003 | 0.498 ± 0.247 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

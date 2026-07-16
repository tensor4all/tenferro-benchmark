# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20260716_100934`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20260716_100934`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260716_100934/cpu_ops_t4_20260716_100934.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260716_100934/linalg_jvp_vjp_t4_20260716_100934.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.875 ± 0.207 | 9.330 ± 0.425 | 29.603 ± 0.340 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 55.186 ± 0.238 | 27.444 ± 0.319 | 88.491 ± 1.155 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 13.857 ± 0.361 | 6.539 ± 0.329 | 28.735 ± 0.296 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 53.473 ± 0.418 | 20.419 ± 0.131 | 87.502 ± 1.230 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 6.181 ± 0.285 | 4.432 ± 0.038 | 8.390 ± 0.735 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 25.755 ± 0.780 | 24.032 ± 0.094 | 21.056 ± 2.355 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.773 ± 0.110 | 2.834 ± 0.045 | 10.279 ± 1.208 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 24.647 ± 0.290 | 11.612 ± 0.425 | 26.781 ± 1.371 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.070 ± 0.300 | 4.484 ± 0.032 | 13.436 ± 1.250 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 35.758 ± 1.085 | 18.545 ± 0.116 | 45.213 ± 2.752 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.125 ± 0.117 | 4.313 ± 0.028 | 14.295 ± 1.386 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 34.135 ± 1.079 | 18.294 ± 0.110 | 59.749 ± 4.136 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.692 ± 0.022 | 1.578 ± 0.032 | 4.147 ± 0.269 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 6.231 ± 0.087 | 3.268 ± 0.135 | 10.265 ± 0.117 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.234 ± 0.040 | 1.678 ± 0.027 | 5.500 ± 0.159 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.431 ± 0.077 | 4.675 ± 0.098 | 10.894 ± 0.218 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 20.825 ± 0.049 | 17.897 ± 0.271 | 52.259 ± 0.724 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 96.337 ± 0.696 | 76.321 ± 0.480 | 162.473 ± 4.459 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 21.611 ± 0.133 | 16.049 ± 0.158 | 50.482 ± 1.542 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 95.077 ± 0.239 | 62.869 ± 0.170 | 161.060 ± 4.606 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.112 ± 0.003 | 0.334 ± 0.026 | 1.477 ± 0.027 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.116 ± 0.005 | 0.360 ± 0.013 | 1.355 ± 0.018 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.125 ± 0.007 | 0.371 ± 0.013 | 1.542 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.121 ± 0.003 | 0.323 ± 0.015 | 3.551 ± 0.065 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.127 ± 0.007 | 0.334 ± 0.012 | 3.903 ± 0.094 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.135 ± 0.009 | 0.342 ± 0.015 | 4.721 ± 0.133 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.117 ± 0.004 | 0.524 ± 0.025 | 2.015 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.117 ± 0.003 | 0.517 ± 0.028 | 2.056 ± 0.057 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.122 ± 0.007 | 0.522 ± 0.014 | 3.105 ± 0.097 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.120 ± 0.003 | 0.498 ± 0.077 | 5.795 ± 0.400 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.121 ± 0.004 | 0.464 ± 0.031 | 5.730 ± 0.345 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.128 ± 0.002 | 0.464 ± 0.020 | 6.297 ± 2.059 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.316 ± 0.126 | 0.401 ± 0.014 | 3.176 ± 0.037 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.280 ± 0.009 | 0.397 ± 0.012 | 2.581 ± 0.107 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.283 ± 0.013 | 0.406 ± 0.011 | 2.578 ± 0.033 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.594 ± 0.012 | 0.437 ± 0.018 | 6.989 ± 1.332 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.531 ± 0.061 | 0.436 ± 0.021 | 6.452 ± 1.245 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.592 ± 0.019 | 0.438 ± 0.042 | 5.256 ± 1.604 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.073 ± 0.006 | 0.722 ± 0.049 | 2.386 ± 0.065 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.073 ± 0.006 | 0.688 ± 0.083 | 2.341 ± 0.038 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.073 ± 0.007 | 0.696 ± 0.060 | 2.584 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.116 ± 0.008 | 0.464 ± 0.052 | 5.132 ± 0.122 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.116 ± 0.009 | 0.454 ± 0.011 | 4.288 ± 0.855 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.116 ± 0.004 | 0.458 ± 0.014 | 5.177 ± 0.154 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.093 ± 0.006 | 0.433 ± 0.011 | 1.970 ± 0.021 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.073 ± 0.002 | 0.443 ± 0.021 | 1.899 ± 0.031 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.089 ± 0.003 | 0.457 ± 0.012 | 1.800 ± 0.020 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.105 ± 0.004 | 0.349 ± 0.029 | 4.992 ± 0.233 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.083 ± 0.005 | 0.351 ± 0.010 | 5.081 ± 0.210 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.098 ± 0.003 | 0.379 ± 0.015 | 4.721 ± 0.896 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

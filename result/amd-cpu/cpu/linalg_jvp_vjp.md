# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20260716_081717`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20260716_081717`.

- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

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

- CSV: `data/results/amd-cpu/cpu/einsum/20260716_081717/cpu_ops_t4_20260716_081717.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260716_081717/linalg_jvp_vjp_t4_20260716_081717.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 14.349 ± 0.314 | 7.679 ± 0.228 | 29.508 ± 0.288 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 54.466 ± 0.378 | 27.081 ± 0.256 | 87.934 ± 0.916 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 13.529 ± 0.174 | 6.663 ± 0.112 | 28.565 ± 0.634 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 53.019 ± 0.855 | 20.258 ± 0.277 | 87.883 ± 2.625 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 6.396 ± 0.425 | 4.456 ± 0.006 | 7.057 ± 2.102 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 25.893 ± 0.493 | 23.894 ± 0.414 | 23.766 ± 2.318 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.733 ± 0.119 | 2.919 ± 0.106 | 10.291 ± 1.760 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 24.485 ± 0.034 | 11.530 ± 0.047 | 27.751 ± 5.654 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.085 ± 0.635 | 4.366 ± 0.020 | 12.854 ± 0.993 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 36.163 ± 0.597 | 18.448 ± 0.393 | 43.595 ± 0.802 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.266 ± 0.140 | 4.231 ± 0.015 | 13.872 ± 1.407 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 34.074 ± 0.485 | 18.301 ± 0.204 | 56.605 ± 10.465 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.594 ± 0.021 | 1.597 ± 0.030 | 4.157 ± 0.090 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 6.061 ± 0.049 | 3.159 ± 0.180 | 10.320 ± 0.164 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.154 ± 0.056 | 1.696 ± 0.040 | 5.579 ± 0.187 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.334 ± 0.061 | 4.531 ± 0.121 | 11.176 ± 1.702 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 22.354 ± 0.325 | 17.764 ± 0.317 | 52.812 ± 1.935 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 96.461 ± 0.638 | 80.634 ± 6.353 | 164.388 ± 1.464 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 21.732 ± 0.359 | 15.844 ± 0.093 | 50.359 ± 2.140 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 94.478 ± 0.455 | 63.366 ± 0.479 | 158.732 ± 6.097 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.064 ± 0.009 | 0.336 ± 0.028 | 1.377 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.075 ± 0.005 | 0.357 ± 0.018 | 1.480 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.083 ± 0.006 | 0.368 ± 0.010 | 1.508 ± 0.029 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.071 ± 0.017 | 0.321 ± 0.014 | 3.395 ± 0.096 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.076 ± 0.007 | 0.333 ± 0.011 | 3.414 ± 0.100 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.091 ± 0.019 | 0.343 ± 0.009 | 4.013 ± 0.954 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.080 ± 0.012 | 0.508 ± 0.011 | 1.925 ± 0.054 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.085 ± 0.017 | 0.510 ± 0.013 | 2.097 ± 0.049 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.090 ± 0.005 | 0.514 ± 0.017 | 3.026 ± 0.037 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.080 ± 0.008 | 0.463 ± 0.024 | 5.216 ± 0.159 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.080 ± 0.006 | 0.469 ± 0.026 | 5.593 ± 0.327 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.084 ± 0.002 | 0.462 ± 0.021 | 6.917 ± 1.631 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.187 ± 0.015 | 0.398 ± 0.016 | 3.070 ± 0.020 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.208 ± 0.011 | 0.399 ± 0.009 | 2.548 ± 0.054 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.215 ± 0.011 | 0.407 ± 0.015 | 2.255 ± 0.137 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.480 ± 0.006 | 0.443 ± 0.043 | 6.795 ± 1.329 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.475 ± 0.015 | 0.442 ± 0.026 | 6.472 ± 0.999 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.483 ± 0.029 | 0.446 ± 0.023 | 4.715 ± 1.290 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.050 ± 0.007 | 0.694 ± 0.010 | 2.275 ± 0.052 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.052 ± 0.006 | 0.696 ± 0.091 | 2.176 ± 0.130 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.057 ± 0.003 | 0.697 ± 0.058 | 2.308 ± 0.053 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.068 ± 0.007 | 0.453 ± 0.009 | 4.833 ± 0.802 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.074 ± 0.013 | 0.458 ± 0.015 | 4.971 ± 0.734 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.080 ± 0.011 | 0.459 ± 0.013 | 5.022 ± 0.101 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.042 ± 0.003 | 0.428 ± 0.011 | 2.059 ± 0.023 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.058 ± 0.001 | 0.436 ± 0.011 | 1.942 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.078 ± 0.008 | 0.454 ± 0.009 | 1.861 ± 0.015 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.053 ± 0.003 | 0.361 ± 0.015 | 4.868 ± 0.210 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.063 ± 0.003 | 0.354 ± 0.007 | 4.950 ± 0.339 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.079 ± 0.010 | 0.382 ± 0.015 | 4.959 ± 0.181 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

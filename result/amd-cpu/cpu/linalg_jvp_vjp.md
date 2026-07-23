# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20260723_051422`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20260723_051422`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260723_051422/cpu_ops_t4_20260723_051422.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260723_051422/linalg_jvp_vjp_t4_20260723_051422.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.336 ± 0.120 | 7.837 ± 0.846 | 18.600 ± 0.648 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 48.518 ± 2.604 | 29.920 ± 0.636 | 87.844 ± 2.279 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.801 ± 3.070 | 6.676 ± 0.137 | 18.249 ± 0.853 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 47.011 ± 3.105 | 22.631 ± 0.654 | 88.731 ± 1.645 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 8.054 ± 0.223 | 4.366 ± 0.261 | 6.904 ± 2.481 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 28.615 ± 9.108 | 27.721 ± 0.900 | 24.639 ± 3.808 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 7.233 ± 0.126 | 2.758 ± 0.119 | 8.158 ± 0.506 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 31.109 ± 4.542 | 12.960 ± 0.267 | 30.187 ± 8.425 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 10.232 ± 2.514 | 4.604 ± 0.028 | 10.496 ± 1.133 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 31.151 ± 4.750 | 20.848 ± 0.951 | 44.076 ± 1.611 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 10.596 ± 2.667 | 4.448 ± 0.054 | 11.245 ± 1.693 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 30.298 ± 1.471 | 20.361 ± 0.738 | 60.387 ± 4.090 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.180 ± 0.091 | 1.572 ± 0.032 | 3.128 ± 0.631 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 9.088 ± 0.705 | 3.417 ± 0.195 | 10.290 ± 0.657 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.327 ± 0.048 | 1.671 ± 0.017 | 4.978 ± 0.299 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.849 ± 1.381 | 4.933 ± 0.279 | 10.948 ± 0.121 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 24.284 ± 5.945 | 19.567 ± 0.228 | 39.784 ± 1.925 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 106.834 ± 0.890 | 85.594 ± 2.051 | 163.130 ± 6.305 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 23.724 ± 0.184 | 17.360 ± 0.168 | 37.190 ± 4.759 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 103.813 ± 1.362 | 70.554 ± 1.030 | 160.055 ± 5.213 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.112 ± 0.007 | 0.387 ± 0.038 | 1.703 ± 0.108 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.123 ± 0.007 | 0.387 ± 0.060 | 1.900 ± 0.215 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.131 ± 0.011 | 0.414 ± 0.013 | 1.635 ± 0.160 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.123 ± 0.013 | 0.374 ± 0.023 | 4.923 ± 1.305 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.135 ± 0.011 | 0.366 ± 0.019 | 4.174 ± 0.348 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.143 ± 0.008 | 0.396 ± 0.010 | 4.691 ± 0.564 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.115 ± 0.016 | 0.566 ± 0.011 | 2.408 ± 0.277 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.117 ± 0.009 | 0.562 ± 0.012 | 2.537 ± 0.177 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.122 ± 0.008 | 0.570 ± 0.008 | 2.721 ± 0.048 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.124 ± 0.005 | 0.501 ± 0.011 | 6.673 ± 0.454 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.124 ± 0.006 | 0.496 ± 0.009 | 7.560 ± 0.779 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.128 ± 0.003 | 0.505 ± 0.009 | 6.219 ± 0.199 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.284 ± 0.012 | 0.445 ± 0.012 | 3.412 ± 0.211 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.285 ± 0.012 | 0.454 ± 0.016 | 2.856 ± 0.352 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.278 ± 0.015 | 0.446 ± 0.011 | 2.464 ± 0.072 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.592 ± 0.027 | 0.473 ± 0.012 | 7.218 ± 0.584 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.612 ± 0.051 | 0.492 ± 0.012 | 6.738 ± 1.233 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.598 ± 0.037 | 0.485 ± 0.013 | 6.661 ± 1.051 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.073 ± 0.007 | 0.781 ± 0.043 | 2.392 ± 0.468 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.074 ± 0.006 | 0.780 ± 0.039 | 2.195 ± 0.380 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.075 ± 0.003 | 0.830 ± 0.037 | 1.752 ± 0.096 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.104 ± 0.003 | 0.507 ± 0.042 | 5.858 ± 0.334 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.106 ± 0.004 | 0.532 ± 0.018 | 5.438 ± 0.732 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.107 ± 0.008 | 0.533 ± 0.023 | 3.730 ± 0.214 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.069 ± 0.002 | 0.525 ± 0.014 | 1.950 ± 0.145 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.079 ± 0.003 | 0.522 ± 0.031 | 1.478 ± 0.096 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.097 ± 0.011 | 0.536 ± 0.024 | 1.462 ± 0.092 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.084 ± 0.002 | 0.408 ± 0.013 | 6.237 ± 1.208 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.091 ± 0.004 | 0.417 ± 0.016 | 4.825 ± 0.512 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.013 | 0.443 ± 0.027 | 4.552 ± 0.238 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

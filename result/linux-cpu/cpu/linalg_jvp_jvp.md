# Linux CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

## Run Inputs

- Threads 1: timestamp `20260630_101340`, raw run `data/results/linux-cpu/cpu/einsum/20260630_101340`
- Threads 4: timestamp `20260630_101651`, raw run `data/results/linux-cpu/cpu/einsum/20260630_101651`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260630_101340`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_101340`.

- tenferro-rs commit: `38b1c5f2a0f0229336dda751d1033cae3cfc106a`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

#### Thread Environment

- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_101340/cpu_ops_t1_20260630_101340.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_101340/linalg_jvp_vjp_t1_20260630_101340.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 14.108 ± 0.880 | 11.872 ± 0.496 | 35.713 ± 0.531 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 73.811 ± 0.991 | 69.945 ± 0.214 | 114.657 ± 2.104 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 13.737 ± 1.092 | 9.573 ± 0.267 | 34.813 ± 0.953 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 72.160 ± 1.609 | 52.395 ± 0.896 | 115.024 ± 2.672 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.565 ± 0.524 | 9.033 ± 0.366 | 11.329 ± 2.356 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 42.010 ± 4.515 | 66.122 ± 0.465 | 49.272 ± 1.670 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 14.071 ± 0.646 | 5.331 ± 0.233 | 15.710 ± 1.620 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 78.016 ± 6.310 | 37.068 ± 0.384 | 61.229 ± 9.646 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.270 ± 0.382 | 7.677 ± 0.118 | 18.722 ± 1.467 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 53.951 ± 2.586 | 50.060 ± 1.795 | 80.160 ± 1.676 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.837 ± 0.692 | 7.417 ± 0.500 | 19.779 ± 1.667 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 82.998 ± 4.668 | 49.047 ± 0.558 | 99.392 ± 2.731 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.886 ± 0.273 | 1.692 ± 0.058 | 5.162 ± 0.159 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.050 ± 0.437 | 5.562 ± 0.010 | 15.936 ± 0.153 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.480 ± 0.090 | 2.041 ± 0.038 | 6.026 ± 0.422 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 9.790 ± 0.448 | 8.920 ± 0.008 | 16.611 ± 0.128 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 112.985 ± 6.799 | 21.555 ± 0.108 | 61.314 ± 1.072 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 1308.396 ± 22.184 | 139.522 ± 1.127 | 205.729 ± 2.647 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 115.887 ± 2.918 | 17.998 ± 0.654 | 60.386 ± 1.963 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 1317.331 ± 29.916 | 112.787 ± 1.273 | 202.237 ± 6.769 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.003 | 0.323 ± 0.010 | 1.397 ± 0.050 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.036 ± 0.002 | 0.331 ± 0.009 | 1.400 ± 0.066 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.046 ± 0.002 | 0.343 ± 0.011 | 1.586 ± 0.057 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.032 ± 0.005 | 0.321 ± 0.020 | 3.291 ± 0.364 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.040 ± 0.002 | 0.326 ± 0.009 | 3.407 ± 0.512 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.048 ± 0.002 | 0.335 ± 0.012 | 4.143 ± 1.221 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.058 ± 0.002 | 0.449 ± 0.010 | 1.835 ± 0.070 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.063 ± 0.002 | 0.454 ± 0.011 | 1.957 ± 0.095 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.063 ± 0.002 | 0.464 ± 0.019 | 2.911 ± 0.639 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.080 ± 0.002 | 0.426 ± 0.027 | 4.584 ± 1.267 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.083 ± 0.003 | 0.406 ± 0.010 | 4.645 ± 1.265 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.092 ± 0.002 | 0.413 ± 0.009 | 4.915 ± 2.557 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.088 ± 0.010 | 0.361 ± 0.013 | 2.957 ± 0.313 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.091 ± 0.002 | 0.358 ± 0.010 | 2.489 ± 0.124 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.096 ± 0.002 | 0.366 ± 0.017 | 2.281 ± 0.111 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.155 ± 0.009 | 0.393 ± 0.010 | 4.866 ± 2.573 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.168 ± 0.002 | 0.400 ± 0.016 | 4.652 ± 1.727 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.191 ± 0.025 | 0.405 ± 0.042 | 4.744 ± 2.472 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.021 ± 0.001 | 0.687 ± 0.006 | 2.074 ± 0.070 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.020 ± 0.001 | 0.691 ± 0.012 | 2.115 ± 0.082 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.002 | 0.695 ± 0.011 | 2.447 ± 0.109 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.030 ± 0.002 | 0.444 ± 0.011 | 4.254 ± 1.188 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.030 ± 0.002 | 0.444 ± 0.011 | 4.220 ± 1.276 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.001 | 0.449 ± 0.012 | 3.642 ± 1.384 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.021 ± 0.001 | 0.430 ± 0.013 | 1.681 ± 0.075 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.027 ± 0.004 | 0.433 ± 0.011 | 1.560 ± 0.040 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.038 ± 0.002 | 0.453 ± 0.010 | 1.789 ± 0.084 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.040 ± 0.002 | 0.350 ± 0.016 | 4.076 ± 1.441 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.045 ± 0.001 | 0.355 ± 0.061 | 4.002 ± 1.311 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.057 ± 0.002 | 0.372 ± 0.015 | 4.086 ± 1.858 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260630_101651`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_101651`.

- tenferro-rs commit: `38b1c5f2a0f0229336dda751d1033cae3cfc106a`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-openblas`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `openblas`
- BLAS version: `OpenBLAS 0.3.26`
- BLAS root: `/opt/openblas`
- BLAS library: `/opt/openblas/lib/libopenblas.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_101651/cpu_ops_t4_20260630_101651.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_101651/linalg_jvp_vjp_t4_20260630_101651.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.504 ± 0.406 | 7.595 ± 0.926 | 29.718 ± 0.548 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 56.850 ± 0.946 | 28.978 ± 0.691 | 88.928 ± 1.799 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 15.460 ± 0.193 | 6.353 ± 0.250 | 28.668 ± 0.550 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 54.228 ± 0.767 | 21.462 ± 0.474 | 89.025 ± 2.061 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.842 ± 1.099 | 4.199 ± 0.213 | 8.078 ± 1.635 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 35.730 ± 1.965 | 25.244 ± 0.931 | 21.637 ± 3.874 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 16.612 ± 1.119 | 2.688 ± 0.100 | 10.786 ± 1.193 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 58.810 ± 5.060 | 12.010 ± 0.328 | 32.347 ± 2.345 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.841 ± 0.301 | 4.639 ± 0.079 | 13.850 ± 1.091 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 35.500 ± 2.553 | 19.490 ± 0.851 | 45.028 ± 2.110 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 14.531 ± 0.991 | 4.465 ± 0.183 | 14.178 ± 0.367 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 50.373 ± 4.164 | 19.182 ± 0.501 | 60.427 ± 3.065 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.139 ± 0.048 | 1.610 ± 0.063 | 4.144 ± 0.049 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 8.906 ± 0.088 | 3.225 ± 0.092 | 10.425 ± 0.105 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.851 ± 0.061 | 1.696 ± 0.017 | 5.481 ± 0.588 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 11.717 ± 0.258 | 4.729 ± 0.191 | 11.128 ± 0.333 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 120.287 ± 6.927 | 18.413 ± 0.334 | 52.723 ± 1.285 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 1298.708 ± 41.768 | 79.541 ± 1.718 | 162.087 ± 5.363 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 118.231 ± 4.435 | 16.467 ± 0.559 | 51.149 ± 1.111 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 1274.155 ± 12.973 | 65.903 ± 0.996 | 163.106 ± 2.989 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.058 ± 0.016 | 0.345 ± 0.077 | 1.430 ± 0.056 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.072 ± 0.016 | 0.375 ± 0.084 | 1.448 ± 0.066 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.087 ± 0.019 | 0.358 ± 0.046 | 1.425 ± 0.065 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.059 ± 0.006 | 0.318 ± 0.011 | 3.164 ± 0.383 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.062 ± 0.007 | 0.333 ± 0.076 | 3.191 ± 0.291 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.077 ± 0.006 | 0.341 ± 0.022 | 4.037 ± 0.894 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.114 ± 0.004 | 0.553 ± 0.088 | 1.822 ± 0.061 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.118 ± 0.004 | 0.531 ± 0.033 | 1.937 ± 0.068 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.123 ± 0.011 | 0.544 ± 0.052 | 3.152 ± 0.222 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.155 ± 0.011 | 0.471 ± 0.022 | 4.599 ± 1.130 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.162 ± 0.010 | 0.471 ± 0.037 | 4.648 ± 1.258 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.194 ± 0.037 | 0.468 ± 0.024 | 4.936 ± 2.555 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.178 ± 0.013 | 0.408 ± 0.044 | 2.777 ± 0.256 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.178 ± 0.010 | 0.410 ± 0.028 | 2.303 ± 0.078 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.182 ± 0.020 | 0.439 ± 0.042 | 2.159 ± 0.045 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.295 ± 0.016 | 0.440 ± 0.064 | 4.814 ± 2.706 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.308 ± 0.016 | 0.438 ± 0.015 | 4.691 ± 1.680 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.330 ± 0.011 | 0.445 ± 0.025 | 4.865 ± 2.665 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.043 ± 0.003 | 0.693 ± 0.014 | 2.100 ± 0.109 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.045 ± 0.007 | 0.760 ± 0.149 | 2.239 ± 0.097 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.053 ± 0.006 | 0.723 ± 0.074 | 2.023 ± 0.058 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.073 ± 0.007 | 0.449 ± 0.037 | 4.020 ± 1.202 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.073 ± 0.006 | 0.453 ± 0.015 | 4.156 ± 1.171 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.087 ± 0.013 | 0.455 ± 0.011 | 4.293 ± 1.186 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.040 ± 0.002 | 0.438 ± 0.020 | 1.771 ± 0.082 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.051 ± 0.003 | 0.444 ± 0.009 | 1.675 ± 0.014 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.064 ± 0.003 | 0.464 ± 0.010 | 1.767 ± 0.045 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.073 ± 0.004 | 0.352 ± 0.058 | 4.074 ± 1.364 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.081 ± 0.004 | 0.349 ± 0.010 | 3.932 ± 1.293 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.096 ± 0.003 | 0.379 ± 0.015 | 4.249 ± 1.781 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

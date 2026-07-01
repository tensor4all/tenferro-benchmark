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

- Threads 1: timestamp `20260701_094811`, raw run `data/results/linux-cpu/cpu/einsum/20260701_094811`
- Threads 4: timestamp `20260701_095109`, raw run `data/results/linux-cpu/cpu/einsum/20260701_095109`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260701_094811`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260701_094811`.

- tenferro-rs commit: `c6a7d6265b2313e03208dc94d62b9efc1f0d8c62`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260701_094811/cpu_ops_t1_20260701_094811.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260701_094811/linalg_jvp_vjp_t1_20260701_094811.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 13.767 ± 0.551 | 11.589 ± 0.006 | 35.542 ± 0.363 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 72.295 ± 1.014 | 70.654 ± 1.924 | 115.001 ± 0.932 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 13.107 ± 0.416 | 9.717 ± 0.117 | 34.777 ± 0.998 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 71.119 ± 1.073 | 52.033 ± 0.517 | 116.206 ± 1.626 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.481 ± 0.771 | 8.951 ± 0.069 | 10.791 ± 2.384 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 41.422 ± 4.295 | 67.428 ± 14.524 | 44.890 ± 4.878 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.336 ± 0.682 | 5.200 ± 0.046 | 14.715 ± 1.167 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 42.858 ± 3.159 | 37.252 ± 0.211 | 60.692 ± 10.583 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.108 ± 0.499 | 7.593 ± 0.050 | 17.838 ± 1.044 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 53.301 ± 3.199 | 50.715 ± 1.033 | 79.323 ± 9.724 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 8.865 ± 0.663 | 7.323 ± 0.063 | 19.436 ± 0.940 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 53.543 ± 0.951 | 49.427 ± 0.568 | 99.794 ± 0.862 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.760 ± 1.028 | 1.692 ± 0.005 | 5.089 ± 0.114 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.878 ± 0.320 | 5.609 ± 0.323 | 16.011 ± 0.094 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.361 ± 0.041 | 2.049 ± 0.008 | 5.851 ± 0.388 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 8.956 ± 0.311 | 8.883 ± 0.007 | 16.586 ± 0.213 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 23.036 ± 1.046 | 21.603 ± 0.566 | 61.478 ± 1.979 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 133.784 ± 5.309 | 139.706 ± 2.842 | 207.740 ± 4.001 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 25.457 ± 1.708 | 18.142 ± 0.099 | 59.475 ± 2.073 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 145.059 ± 10.585 | 113.328 ± 0.998 | 202.582 ± 5.547 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.041 ± 0.001 | 0.330 ± 0.009 | 1.259 ± 0.035 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.036 ± 0.001 | 0.340 ± 0.010 | 1.455 ± 0.070 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.043 ± 0.004 | 0.345 ± 0.010 | 1.602 ± 0.060 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.044 ± 0.002 | 0.314 ± 0.009 | 3.257 ± 0.371 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.038 ± 0.002 | 0.327 ± 0.008 | 3.385 ± 0.307 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.045 ± 0.006 | 0.337 ± 0.010 | 4.129 ± 1.266 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.079 ± 0.007 | 0.451 ± 0.012 | 1.987 ± 0.057 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.060 ± 0.001 | 0.455 ± 0.010 | 1.958 ± 0.051 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.066 ± 0.006 | 0.459 ± 0.011 | 3.205 ± 0.611 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.084 ± 0.005 | 0.406 ± 0.013 | 4.573 ± 1.263 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.075 ± 0.022 | 0.410 ± 0.009 | 4.631 ± 1.274 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.066 ± 0.003 | 0.416 ± 0.008 | 4.991 ± 2.606 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.123 ± 0.004 | 0.364 ± 0.017 | 2.945 ± 0.377 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.093 ± 0.002 | 0.364 ± 0.017 | 2.367 ± 0.150 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.121 ± 0.032 | 0.366 ± 0.010 | 2.481 ± 0.028 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.086 ± 0.002 | 0.387 ± 0.013 | 4.947 ± 2.760 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.066 ± 0.002 | 0.399 ± 0.012 | 5.154 ± 1.743 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.080 ± 0.014 | 0.399 ± 0.008 | 4.976 ± 2.688 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.029 ± 0.001 | 0.687 ± 0.010 | 2.058 ± 0.108 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.023 ± 0.002 | 0.689 ± 0.006 | 2.225 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.001 | 0.693 ± 0.011 | 2.197 ± 0.441 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.040 ± 0.002 | 0.445 ± 0.011 | 4.166 ± 1.028 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.036 ± 0.012 | 0.472 ± 0.050 | 3.712 ± 1.461 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.031 ± 0.002 | 0.451 ± 0.009 | 3.424 ± 1.041 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.002 | 0.438 ± 0.012 | 1.578 ± 0.064 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.030 ± 0.003 | 0.443 ± 0.011 | 1.701 ± 0.009 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.049 ± 0.004 | 0.460 ± 0.011 | 1.774 ± 0.068 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.057 ± 0.003 | 0.349 ± 0.015 | 3.628 ± 1.312 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.049 ± 0.001 | 0.356 ± 0.015 | 4.000 ± 1.298 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.080 ± 0.020 | 0.372 ± 0.008 | 4.244 ± 1.834 |

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
- Timestamp: `20260701_095109`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260701_095109`.

- tenferro-rs commit: `c6a7d6265b2313e03208dc94d62b9efc1f0d8c62`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260701_095109/cpu_ops_t4_20260701_095109.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260701_095109/linalg_jvp_vjp_t4_20260701_095109.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.220 ± 0.268 | 7.822 ± 0.296 | 29.573 ± 0.610 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 54.496 ± 1.099 | 27.546 ± 0.163 | 88.416 ± 0.567 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 12.925 ± 0.122 | 6.613 ± 0.095 | 28.609 ± 0.516 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 53.811 ± 8.371 | 20.804 ± 0.565 | 87.767 ± 1.117 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.121 ± 1.361 | 4.445 ± 0.065 | 8.232 ± 2.103 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 36.007 ± 2.482 | 23.873 ± 0.526 | 21.080 ± 3.652 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 10.415 ± 1.060 | 2.923 ± 0.080 | 11.575 ± 1.823 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 36.442 ± 1.571 | 11.482 ± 0.062 | 27.243 ± 2.968 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.881 ± 0.181 | 6.384 ± 0.277 | 13.600 ± 0.640 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 33.994 ± 6.835 | 18.571 ± 0.230 | 45.137 ± 2.073 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 7.966 ± 0.051 | 4.847 ± 1.222 | 14.583 ± 2.681 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 33.695 ± 0.533 | 18.320 ± 0.161 | 61.677 ± 3.643 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.157 ± 0.050 | 1.537 ± 0.042 | 4.068 ± 0.050 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 8.855 ± 0.083 | 3.120 ± 0.060 | 10.065 ± 0.252 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.542 ± 0.074 | 1.611 ± 0.018 | 5.352 ± 0.716 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 7.583 ± 0.271 | 4.492 ± 0.040 | 11.008 ± 0.176 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 20.987 ± 0.192 | 18.216 ± 0.309 | 52.298 ± 1.489 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 95.985 ± 1.595 | 78.058 ± 2.799 | 164.092 ± 4.000 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 26.188 ± 3.065 | 16.258 ± 0.201 | 50.266 ± 1.605 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 105.820 ± 3.469 | 63.779 ± 1.158 | 161.666 ± 3.406 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.055 ± 0.010 | 0.329 ± 0.014 | 1.334 ± 0.054 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.067 ± 0.016 | 0.360 ± 0.047 | 1.374 ± 0.048 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.083 ± 0.011 | 0.359 ± 0.037 | 1.491 ± 0.054 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.060 ± 0.018 | 0.319 ± 0.011 | 3.193 ± 0.335 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.068 ± 0.010 | 0.329 ± 0.009 | 3.171 ± 0.315 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.097 ± 0.019 | 0.338 ± 0.013 | 4.170 ± 0.853 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.120 ± 0.009 | 0.519 ± 0.019 | 1.748 ± 0.078 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.128 ± 0.007 | 0.514 ± 0.012 | 1.882 ± 0.078 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.134 ± 0.006 | 0.522 ± 0.037 | 3.091 ± 0.237 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.109 ± 0.004 | 0.466 ± 0.039 | 4.500 ± 1.292 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.120 ± 0.011 | 0.462 ± 0.013 | 4.621 ± 1.476 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.129 ± 0.015 | 0.468 ± 0.023 | 5.004 ± 2.799 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.171 ± 0.006 | 0.406 ± 0.031 | 2.849 ± 0.232 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.177 ± 0.008 | 0.426 ± 0.062 | 2.241 ± 0.122 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.202 ± 0.022 | 0.416 ± 0.017 | 2.112 ± 0.064 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.116 ± 0.007 | 0.437 ± 0.026 | 4.791 ± 2.585 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.117 ± 0.010 | 0.471 ± 0.033 | 4.594 ± 1.628 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.136 ± 0.022 | 0.443 ± 0.018 | 4.830 ± 2.704 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.047 ± 0.006 | 0.694 ± 0.007 | 2.113 ± 0.090 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.049 ± 0.004 | 0.696 ± 0.008 | 1.759 ± 0.191 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.053 ± 0.009 | 0.697 ± 0.011 | 2.328 ± 0.056 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.078 ± 0.004 | 0.450 ± 0.011 | 4.034 ± 1.043 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.083 ± 0.014 | 0.452 ± 0.010 | 3.956 ± 0.867 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.088 ± 0.010 | 0.454 ± 0.009 | 4.255 ± 1.084 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.045 ± 0.009 | 0.437 ± 0.011 | 1.717 ± 0.034 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.051 ± 0.001 | 0.449 ± 0.029 | 1.753 ± 0.072 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.071 ± 0.006 | 0.459 ± 0.011 | 1.665 ± 0.018 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.076 ± 0.006 | 0.351 ± 0.018 | 3.988 ± 1.314 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.083 ± 0.004 | 0.383 ± 0.041 | 4.060 ± 1.467 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.108 ± 0.006 | 0.370 ± 0.010 | 3.907 ± 1.797 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

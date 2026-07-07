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

- Threads 1: timestamp `20260707_025543`, raw run `data/results/linux-cpu/cpu/einsum/20260707_025543`
- Threads 4: timestamp `20260707_030612`, raw run `data/results/linux-cpu/cpu/einsum/20260707_030612`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260707_025543`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260707_025543`.

- tenferro-rs commit: `ee3c70c57e0edf395a88fcb515508ebcd2109b41`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260707_025543/cpu_ops_t1_20260707_025543.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260707_025543/linalg_jvp_vjp_t1_20260707_025543.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 14.324 ± 0.848 | 12.790 ± 0.638 | 34.011 ± 0.597 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 77.340 ± 1.170 | 75.463 ± 1.488 | 112.013 ± 1.056 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 21.090 ± 2.636 | 10.136 ± 0.169 | 33.283 ± 0.327 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 94.565 ± 1.967 | 58.488 ± 8.792 | 112.646 ± 1.928 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.497 ± 0.084 | 9.851 ± 0.772 | 11.304 ± 1.840 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 45.066 ± 4.839 | 74.107 ± 1.937 | 46.304 ± 0.522 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 15.399 ± 0.762 | 5.492 ± 0.122 | 14.868 ± 1.573 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 81.297 ± 4.434 | 41.376 ± 1.265 | 58.158 ± 4.399 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.587 ± 0.431 | 7.974 ± 0.666 | 17.066 ± 1.216 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 56.854 ± 9.070 | 55.313 ± 1.058 | 65.442 ± 6.774 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.974 ± 1.202 | 7.861 ± 0.542 | 17.831 ± 0.974 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 88.299 ± 4.440 | 53.467 ± 1.486 | 93.932 ± 3.361 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.665 ± 0.090 | 1.806 ± 0.067 | 4.685 ± 0.042 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.515 ± 0.645 | 5.894 ± 0.200 | 14.935 ± 0.165 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.334 ± 0.096 | 2.145 ± 0.045 | 5.119 ± 0.238 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.252 ± 0.087 | 9.968 ± 0.731 | 15.181 ± 0.133 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 24.466 ± 1.278 | 22.782 ± 0.567 | 58.663 ± 1.492 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 148.660 ± 4.470 | 151.736 ± 3.025 | 196.703 ± 4.303 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 26.546 ± 0.895 | 19.251 ± 0.657 | 56.894 ± 0.971 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 162.520 ± 12.048 | 122.698 ± 2.955 | 195.850 ± 3.281 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.001 | 0.354 ± 0.029 | 1.040 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.038 ± 0.002 | 0.371 ± 0.059 | 1.063 ± 0.032 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.045 ± 0.004 | 0.378 ± 0.056 | 1.432 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.059 ± 0.005 | 0.331 ± 0.027 | 3.100 ± 0.359 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.069 ± 0.008 | 0.353 ± 0.029 | 3.121 ± 0.213 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.073 ± 0.003 | 0.368 ± 0.021 | 3.787 ± 1.068 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.064 ± 0.007 | 0.519 ± 0.035 | 1.641 ± 0.015 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.067 ± 0.005 | 0.482 ± 0.045 | 1.664 ± 0.040 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.070 ± 0.006 | 0.524 ± 0.067 | 2.276 ± 0.029 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.087 ± 0.010 | 0.484 ± 0.094 | 4.059 ± 1.355 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.088 ± 0.005 | 0.435 ± 0.050 | 4.381 ± 1.356 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.093 ± 0.014 | 0.438 ± 0.018 | 4.478 ± 1.552 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.101 ± 0.009 | 0.389 ± 0.025 | 2.701 ± 0.180 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.097 ± 0.010 | 0.385 ± 0.030 | 2.002 ± 0.022 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.107 ± 0.017 | 0.390 ± 0.018 | 2.263 ± 0.063 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.166 ± 0.014 | 0.420 ± 0.037 | 4.277 ± 2.556 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.189 ± 0.032 | 0.432 ± 0.039 | 4.383 ± 1.820 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.186 ± 0.028 | 0.423 ± 0.016 | 4.517 ± 2.593 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.023 ± 0.001 | 0.807 ± 0.041 | 1.779 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.024 ± 0.003 | 0.738 ± 0.038 | 2.002 ± 0.026 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.026 ± 0.001 | 0.727 ± 0.022 | 2.035 ± 0.025 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.030 ± 0.002 | 0.547 ± 0.082 | 3.844 ± 1.139 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.032 ± 0.002 | 0.501 ± 0.020 | 3.883 ± 1.039 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.034 ± 0.002 | 0.473 ± 0.020 | 4.078 ± 1.081 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.028 ± 0.004 | 0.458 ± 0.030 | 1.478 ± 0.026 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.031 ± 0.003 | 0.472 ± 0.018 | 1.726 ± 0.031 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.045 ± 0.003 | 0.494 ± 0.047 | 1.454 ± 0.019 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.042 ± 0.010 | 0.368 ± 0.032 | 3.792 ± 1.295 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.050 ± 0.003 | 0.371 ± 0.024 | 3.633 ± 1.185 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.062 ± 0.005 | 0.393 ± 0.030 | 3.858 ± 1.452 |

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
- Timestamp: `20260707_030612`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260707_030612`.

- tenferro-rs commit: `ee3c70c57e0edf395a88fcb515508ebcd2109b41`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260707_030612/cpu_ops_t4_20260707_030612.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260707_030612/linalg_jvp_vjp_t4_20260707_030612.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 14.195 ± 2.106 | 7.913 ± 0.207 | 29.610 ± 0.467 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 58.217 ± 0.793 | 29.509 ± 1.439 | 88.196 ± 2.555 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 16.918 ± 0.728 | 6.789 ± 0.264 | 28.711 ± 0.416 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 68.994 ± 4.782 | 21.990 ± 0.582 | 89.248 ± 2.400 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.305 ± 1.309 | 4.554 ± 0.101 | 8.444 ± 1.623 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 36.649 ± 2.037 | 25.897 ± 0.744 | 25.056 ± 0.675 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 15.152 ± 0.385 | 2.865 ± 0.207 | 10.193 ± 0.573 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 55.721 ± 5.330 | 12.853 ± 1.119 | 27.862 ± 2.493 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.887 ± 0.753 | 4.517 ± 0.102 | 14.114 ± 0.698 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 34.972 ± 4.370 | 19.877 ± 0.576 | 44.450 ± 1.918 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 14.478 ± 5.663 | 4.342 ± 0.079 | 14.688 ± 1.356 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 49.327 ± 5.327 | 19.787 ± 0.670 | 61.532 ± 4.157 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.871 ± 0.066 | 1.611 ± 0.019 | 4.075 ± 0.078 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 7.710 ± 0.068 | 3.337 ± 0.100 | 10.478 ± 0.071 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.508 ± 0.079 | 1.696 ± 0.014 | 5.123 ± 0.318 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 6.260 ± 0.405 | 4.791 ± 0.190 | 10.822 ± 0.120 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 21.661 ± 0.387 | 18.765 ± 0.485 | 52.774 ± 1.854 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 99.414 ± 2.496 | 82.054 ± 1.257 | 165.413 ± 3.254 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 22.694 ± 0.239 | 16.867 ± 0.308 | 52.072 ± 1.803 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 103.291 ± 3.927 | 66.541 ± 1.018 | 163.223 ± 3.683 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.091 ± 0.004 | 0.346 ± 0.030 | 1.054 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.100 ± 0.038 | 0.360 ± 0.036 | 1.102 ± 0.024 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.076 ± 0.006 | 0.416 ± 0.057 | 1.138 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.174 ± 0.010 | 0.327 ± 0.032 | 3.167 ± 0.570 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.116 ± 0.008 | 0.344 ± 0.023 | 3.154 ± 0.119 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.130 ± 0.018 | 0.370 ± 0.042 | 3.913 ± 1.142 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.173 ± 0.008 | 0.542 ± 0.029 | 1.667 ± 0.029 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.114 ± 0.010 | 0.545 ± 0.027 | 1.848 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.119 ± 0.004 | 0.552 ± 0.024 | 2.706 ± 0.225 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.257 ± 0.017 | 0.485 ± 0.013 | 3.961 ± 1.316 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.157 ± 0.007 | 0.488 ± 0.016 | 3.866 ± 1.205 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.166 ± 0.009 | 0.516 ± 0.013 | 4.002 ± 1.470 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.279 ± 0.011 | 0.423 ± 0.016 | 2.521 ± 0.037 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.286 ± 0.015 | 0.422 ± 0.013 | 2.275 ± 0.027 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.290 ± 0.054 | 0.428 ± 0.026 | 2.232 ± 0.023 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.533 ± 0.010 | 0.467 ± 0.050 | 3.732 ± 1.348 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.541 ± 0.015 | 0.464 ± 0.014 | 4.808 ± 1.751 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.329 ± 0.009 | 0.488 ± 0.028 | 4.503 ± 2.552 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.080 ± 0.004 | 0.731 ± 0.053 | 1.963 ± 0.035 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.045 ± 0.004 | 0.727 ± 0.016 | 1.768 ± 0.035 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.047 ± 0.005 | 0.735 ± 0.136 | 2.025 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.146 ± 0.006 | 0.467 ± 0.019 | 3.863 ± 1.384 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.077 ± 0.009 | 0.470 ± 0.020 | 3.966 ± 1.115 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.083 ± 0.009 | 0.487 ± 0.081 | 4.069 ± 1.062 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.077 ± 0.003 | 0.490 ± 0.085 | 1.501 ± 0.019 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.087 ± 0.003 | 0.453 ± 0.031 | 1.651 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.119 ± 0.011 | 0.476 ± 0.032 | 1.559 ± 0.024 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.139 ± 0.011 | 0.372 ± 0.034 | 2.892 ± 1.162 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.141 ± 0.016 | 0.375 ± 0.038 | 3.784 ± 1.187 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.174 ± 0.010 | 0.385 ± 0.027 | 3.751 ± 1.335 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

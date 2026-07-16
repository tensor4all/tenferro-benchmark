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

- Threads 1: timestamp `20260716_074016`, raw run `data/results/linux-cpu/cpu/einsum/20260716_074016`
- Threads 4: timestamp `20260716_075030`, raw run `data/results/linux-cpu/cpu/einsum/20260716_075030`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260716_074016`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260716_074016`.

- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_074016/cpu_ops_t1_20260716_074016.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_074016/linalg_jvp_vjp_t1_20260716_074016.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 13.173 ± 0.145 | 11.550 ± 0.361 | 35.251 ± 0.939 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 72.582 ± 0.828 | 69.403 ± 0.236 | 114.581 ± 1.003 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.653 ± 0.050 | 9.712 ± 0.024 | 35.038 ± 0.734 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 70.468 ± 0.558 | 51.675 ± 0.391 | 115.771 ± 2.475 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 6.906 ± 0.085 | 8.957 ± 0.016 | 10.621 ± 2.498 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 36.322 ± 4.816 | 66.259 ± 0.566 | 48.942 ± 1.093 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 6.737 ± 0.085 | 5.218 ± 0.063 | 15.467 ± 1.344 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 38.248 ± 3.856 | 37.249 ± 0.148 | 54.750 ± 6.996 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.012 ± 0.113 | 7.572 ± 0.010 | 17.974 ± 1.249 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 53.143 ± 3.321 | 50.273 ± 0.220 | 80.082 ± 11.324 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 8.911 ± 0.120 | 7.339 ± 0.006 | 19.477 ± 0.511 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 53.210 ± 0.873 | 49.386 ± 0.364 | 98.811 ± 1.167 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.437 ± 0.023 | 1.683 ± 0.010 | 5.143 ± 0.072 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.127 ± 0.075 | 5.847 ± 0.381 | 15.835 ± 0.132 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.002 ± 0.005 | 2.050 ± 0.005 | 5.772 ± 0.355 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.316 ± 0.010 | 8.865 ± 0.013 | 16.428 ± 0.411 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 22.755 ± 0.131 | 21.744 ± 0.352 | 61.971 ± 1.147 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 131.734 ± 0.522 | 139.165 ± 0.849 | 206.434 ± 3.512 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 22.404 ± 0.149 | 18.004 ± 0.112 | 59.592 ± 1.619 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 132.766 ± 1.948 | 112.559 ± 1.080 | 203.404 ± 3.567 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.001 | 0.329 ± 0.011 | 1.373 ± 0.053 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.037 ± 0.001 | 0.335 ± 0.011 | 1.363 ± 0.054 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.048 ± 0.001 | 0.342 ± 0.011 | 1.376 ± 0.048 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.035 ± 0.002 | 0.322 ± 0.009 | 2.990 ± 0.182 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.041 ± 0.001 | 0.329 ± 0.011 | 3.220 ± 0.174 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.052 ± 0.001 | 0.345 ± 0.014 | 4.073 ± 1.104 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.003 | 0.456 ± 0.012 | 1.714 ± 0.059 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.034 ± 0.002 | 0.456 ± 0.013 | 1.860 ± 0.048 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.041 ± 0.002 | 0.490 ± 0.031 | 3.016 ± 0.103 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.031 ± 0.002 | 0.411 ± 0.010 | 4.305 ± 0.506 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.033 ± 0.002 | 0.413 ± 0.009 | 4.463 ± 1.060 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.039 ± 0.002 | 0.446 ± 0.063 | 4.849 ± 2.392 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.101 ± 0.002 | 0.368 ± 0.025 | 3.089 ± 0.344 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.100 ± 0.001 | 0.361 ± 0.010 | 2.255 ± 0.094 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.107 ± 0.002 | 0.365 ± 0.012 | 2.693 ± 0.047 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.249 ± 0.017 | 0.401 ± 0.012 | 4.802 ± 2.711 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.247 ± 0.002 | 0.405 ± 0.012 | 5.016 ± 2.019 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.250 ± 0.013 | 0.406 ± 0.011 | 4.589 ± 1.266 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.022 ± 0.001 | 0.689 ± 0.009 | 1.960 ± 0.028 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.023 ± 0.001 | 0.690 ± 0.009 | 2.162 ± 0.088 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.026 ± 0.001 | 0.732 ± 0.087 | 2.140 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.030 ± 0.001 | 0.452 ± 0.006 | 4.055 ± 1.077 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.030 ± 0.000 | 0.455 ± 0.012 | 4.212 ± 1.046 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.034 ± 0.001 | 0.456 ± 0.009 | 3.995 ± 1.117 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.025 ± 0.001 | 0.430 ± 0.015 | 1.744 ± 0.076 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.030 ± 0.001 | 0.434 ± 0.009 | 1.984 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.043 ± 0.002 | 0.452 ± 0.010 | 1.639 ± 0.024 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.028 ± 0.001 | 0.380 ± 0.018 | 3.922 ± 1.315 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.033 ± 0.003 | 0.355 ± 0.012 | 4.049 ± 1.429 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.045 ± 0.002 | 0.375 ± 0.011 | 4.203 ± 1.709 |

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
- Timestamp: `20260716_075030`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260716_075030`.

- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_075030/cpu_ops_t4_20260716_075030.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_075030/linalg_jvp_vjp_t4_20260716_075030.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.227 ± 0.171 | 7.586 ± 1.008 | 29.362 ± 0.373 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 54.173 ± 1.181 | 27.168 ± 0.386 | 88.442 ± 1.110 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 12.797 ± 0.086 | 6.157 ± 0.240 | 28.630 ± 0.441 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 52.833 ± 0.523 | 20.301 ± 0.164 | 89.262 ± 0.976 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 6.092 ± 0.221 | 3.978 ± 0.054 | 8.437 ± 1.758 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 20.213 ± 5.725 | 23.458 ± 0.199 | 24.695 ± 1.206 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.828 ± 0.572 | 2.573 ± 0.034 | 10.836 ± 3.529 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 21.681 ± 4.164 | 11.523 ± 0.202 | 33.937 ± 3.085 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.937 ± 0.189 | 4.460 ± 0.048 | 14.460 ± 0.444 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 33.157 ± 4.782 | 18.370 ± 0.159 | 44.317 ± 2.478 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 9.087 ± 0.350 | 3.928 ± 0.444 | 14.137 ± 0.989 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 33.074 ± 0.517 | 18.339 ± 0.382 | 61.090 ± 3.537 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.584 ± 0.031 | 1.562 ± 0.046 | 4.130 ± 0.075 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 6.048 ± 0.077 | 3.138 ± 0.055 | 10.430 ± 0.127 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.151 ± 0.027 | 1.665 ± 0.035 | 5.594 ± 0.793 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.280 ± 0.077 | 4.511 ± 0.052 | 10.918 ± 0.360 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 20.894 ± 0.212 | 18.030 ± 0.071 | 52.061 ± 1.512 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 93.744 ± 0.768 | 75.943 ± 0.693 | 164.468 ± 2.281 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.384 ± 0.126 | 16.076 ± 0.120 | 50.551 ± 1.047 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 92.653 ± 0.973 | 63.059 ± 0.245 | 162.935 ± 2.371 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.059 ± 0.008 | 0.328 ± 0.014 | 1.346 ± 0.088 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.064 ± 0.006 | 0.339 ± 0.035 | 1.343 ± 0.040 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.076 ± 0.004 | 0.358 ± 0.034 | 1.454 ± 0.061 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.060 ± 0.004 | 0.317 ± 0.009 | 3.075 ± 0.254 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.068 ± 0.005 | 0.328 ± 0.009 | 3.070 ± 0.517 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.082 ± 0.006 | 0.334 ± 0.010 | 3.929 ± 0.978 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.072 ± 0.007 | 0.526 ± 0.036 | 1.815 ± 0.047 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.076 ± 0.010 | 0.505 ± 0.019 | 1.803 ± 0.076 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.080 ± 0.007 | 0.507 ± 0.021 | 3.186 ± 0.232 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.076 ± 0.005 | 0.460 ± 0.024 | 4.512 ± 1.202 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.080 ± 0.007 | 0.452 ± 0.014 | 4.574 ± 1.210 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.080 ± 0.002 | 0.460 ± 0.016 | 4.840 ± 2.227 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.187 ± 0.013 | 0.404 ± 0.030 | 2.968 ± 0.299 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.198 ± 0.022 | 0.401 ± 0.035 | 2.271 ± 0.110 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.204 ± 0.016 | 0.402 ± 0.025 | 2.217 ± 0.079 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.432 ± 0.023 | 0.438 ± 0.028 | 4.803 ± 2.943 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.441 ± 0.026 | 0.436 ± 0.032 | 4.669 ± 1.648 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.439 ± 0.008 | 0.437 ± 0.014 | 4.921 ± 2.521 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.045 ± 0.005 | 0.690 ± 0.011 | 2.039 ± 0.096 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.047 ± 0.005 | 0.692 ± 0.008 | 2.033 ± 0.072 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.049 ± 0.004 | 0.692 ± 0.041 | 2.187 ± 0.106 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.063 ± 0.003 | 0.448 ± 0.009 | 4.107 ± 0.987 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.064 ± 0.007 | 0.449 ± 0.012 | 4.003 ± 1.108 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.065 ± 0.005 | 0.450 ± 0.011 | 4.133 ± 0.921 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.040 ± 0.001 | 0.455 ± 0.049 | 1.645 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.053 ± 0.004 | 0.430 ± 0.011 | 1.741 ± 0.040 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.068 ± 0.002 | 0.446 ± 0.011 | 1.624 ± 0.053 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.052 ± 0.002 | 0.352 ± 0.022 | 3.886 ± 1.565 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.059 ± 0.002 | 0.348 ± 0.008 | 3.996 ± 1.126 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.073 ± 0.003 | 0.366 ± 0.009 | 4.146 ± 1.828 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

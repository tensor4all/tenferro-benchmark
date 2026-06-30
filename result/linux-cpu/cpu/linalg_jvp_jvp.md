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

- Threads 1: timestamp `20260630_112619`, raw run `data/results/linux-cpu/cpu/einsum/20260630_112619`
- Threads 4: timestamp `20260630_112847`, raw run `data/results/linux-cpu/cpu/einsum/20260630_112847`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260630_112619`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_112619`.

- tenferro-rs commit: `54ca79d5bbeefd83d6e87fcec6309c3e6c36adbf`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_112619/cpu_ops_t1_20260630_112619.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_112619/linalg_jvp_vjp_t1_20260630_112619.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 13.396 ± 1.396 | 11.796 ± 0.128 | 35.709 ± 1.745 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 72.839 ± 0.839 | 70.089 ± 1.189 | 113.803 ± 1.193 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.831 ± 0.117 | 9.635 ± 0.220 | 34.525 ± 0.956 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 73.031 ± 9.741 | 51.939 ± 0.542 | 114.794 ± 1.555 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.151 ± 0.105 | 8.972 ± 0.079 | 12.823 ± 2.466 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 42.999 ± 4.377 | 66.409 ± 0.360 | 45.446 ± 4.044 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 13.727 ± 0.098 | 5.183 ± 0.012 | 15.327 ± 1.953 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 81.111 ± 7.059 | 37.408 ± 2.415 | 57.738 ± 7.940 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 8.946 ± 0.210 | 7.599 ± 0.070 | 18.486 ± 1.007 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 52.862 ± 4.655 | 50.760 ± 5.768 | 77.925 ± 9.384 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.072 ± 0.338 | 7.358 ± 0.047 | 19.429 ± 3.678 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 80.148 ± 5.794 | 49.258 ± 0.200 | 98.531 ± 1.957 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.642 ± 0.021 | 1.697 ± 0.011 | 5.080 ± 0.128 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.895 ± 0.563 | 5.558 ± 0.033 | 15.638 ± 0.320 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.420 ± 0.011 | 2.057 ± 0.008 | 5.890 ± 0.498 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 7.589 ± 0.435 | 8.904 ± 0.032 | 15.970 ± 0.773 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 22.991 ± 0.408 | 21.580 ± 1.355 | 61.383 ± 1.924 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 135.987 ± 4.033 | 140.016 ± 2.129 | 206.843 ± 5.084 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 24.896 ± 0.155 | 18.004 ± 0.298 | 60.094 ± 2.235 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 148.160 ± 4.275 | 113.276 ± 1.195 | 204.566 ± 3.146 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.002 | 0.323 ± 0.009 | 1.421 ± 0.057 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.038 ± 0.002 | 0.337 ± 0.029 | 1.474 ± 0.052 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.047 ± 0.002 | 0.339 ± 0.010 | 1.583 ± 0.069 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.037 ± 0.003 | 0.314 ± 0.012 | 3.203 ± 0.421 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.042 ± 0.001 | 0.327 ± 0.007 | 3.250 ± 0.354 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.050 ± 0.001 | 0.333 ± 0.007 | 4.119 ± 0.874 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.063 ± 0.011 | 0.444 ± 0.010 | 1.980 ± 0.095 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.063 ± 0.002 | 0.447 ± 0.015 | 1.977 ± 0.086 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.069 ± 0.002 | 0.452 ± 0.010 | 2.462 ± 0.025 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.085 ± 0.017 | 0.408 ± 0.013 | 4.628 ± 1.292 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.088 ± 0.003 | 0.410 ± 0.009 | 4.614 ± 1.247 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.093 ± 0.003 | 0.411 ± 0.008 | 5.038 ± 2.710 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.105 ± 0.028 | 0.359 ± 0.016 | 2.997 ± 0.352 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.096 ± 0.002 | 0.355 ± 0.011 | 2.332 ± 0.103 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.103 ± 0.002 | 0.385 ± 0.048 | 2.205 ± 0.070 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.186 ± 0.026 | 0.390 ± 0.008 | 4.835 ± 2.564 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.156 ± 0.012 | 0.397 ± 0.014 | 4.850 ± 1.940 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.179 ± 0.002 | 0.397 ± 0.026 | 4.858 ± 2.598 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.020 ± 0.003 | 0.688 ± 0.012 | 2.114 ± 0.101 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.022 ± 0.001 | 0.694 ± 0.007 | 2.044 ± 0.045 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.024 ± 0.001 | 0.688 ± 0.013 | 2.264 ± 0.041 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.032 ± 0.003 | 0.448 ± 0.014 | 4.010 ± 1.243 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.031 ± 0.002 | 0.449 ± 0.010 | 4.152 ± 1.218 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.032 ± 0.001 | 0.450 ± 0.011 | 4.401 ± 1.456 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.024 ± 0.001 | 0.428 ± 0.013 | 1.657 ± 0.039 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.031 ± 0.002 | 0.429 ± 0.011 | 1.766 ± 0.054 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.044 ± 0.002 | 0.450 ± 0.011 | 1.701 ± 0.131 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.043 ± 0.006 | 0.363 ± 0.031 | 4.195 ± 1.839 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.050 ± 0.001 | 0.347 ± 0.009 | 3.981 ± 1.256 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.064 ± 0.002 | 0.397 ± 0.043 | 3.931 ± 1.575 |

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
- Timestamp: `20260630_112847`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_112847`.

- tenferro-rs commit: `54ca79d5bbeefd83d6e87fcec6309c3e6c36adbf`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_112847/cpu_ops_t4_20260630_112847.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_112847/linalg_jvp_vjp_t4_20260630_112847.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.748 ± 0.263 | 7.959 ± 0.483 | 29.463 ± 0.272 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 56.187 ± 1.258 | 28.876 ± 0.866 | 88.229 ± 1.830 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 13.215 ± 0.314 | 6.681 ± 0.138 | 29.028 ± 0.705 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 55.694 ± 1.173 | 23.919 ± 1.287 | 89.700 ± 2.553 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.062 ± 1.141 | 4.600 ± 0.113 | 8.311 ± 2.178 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 36.169 ± 1.676 | 24.388 ± 0.847 | 24.779 ± 3.657 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 12.833 ± 1.388 | 3.052 ± 0.130 | 10.828 ± 1.467 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 55.087 ± 5.286 | 12.214 ± 0.387 | 33.768 ± 6.079 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.857 ± 0.280 | 4.603 ± 0.102 | 14.495 ± 0.409 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 32.698 ± 4.502 | 19.491 ± 0.585 | 45.624 ± 2.083 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 15.216 ± 0.961 | 4.437 ± 0.131 | 14.327 ± 0.746 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 47.314 ± 5.899 | 19.184 ± 1.327 | 60.884 ± 3.379 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.163 ± 0.048 | 1.651 ± 0.085 | 3.964 ± 0.235 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 8.883 ± 1.397 | 3.291 ± 0.167 | 10.468 ± 0.102 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.849 ± 0.069 | 1.718 ± 0.036 | 5.328 ± 0.418 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 6.690 ± 0.342 | 4.588 ± 0.161 | 11.036 ± 0.353 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 21.962 ± 1.123 | 18.456 ± 0.373 | 51.841 ± 1.315 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 103.836 ± 1.460 | 79.392 ± 2.304 | 161.048 ± 6.089 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 23.259 ± 0.613 | 16.493 ± 0.531 | 51.534 ± 1.993 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 104.253 ± 1.616 | 65.671 ± 0.904 | 161.862 ± 3.828 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.058 ± 0.006 | 0.354 ± 0.103 | 1.468 ± 0.058 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.066 ± 0.009 | 0.377 ± 0.067 | 1.410 ± 0.046 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.075 ± 0.004 | 0.367 ± 0.028 | 1.612 ± 0.077 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.064 ± 0.009 | 0.321 ± 0.010 | 3.101 ± 0.307 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.068 ± 0.009 | 0.335 ± 0.013 | 2.835 ± 0.664 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.078 ± 0.004 | 0.339 ± 0.009 | 3.285 ± 0.905 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.120 ± 0.004 | 0.555 ± 0.064 | 1.916 ± 0.097 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.126 ± 0.011 | 0.533 ± 0.057 | 2.078 ± 0.076 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.121 ± 0.006 | 0.537 ± 0.039 | 2.291 ± 0.056 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.168 ± 0.012 | 0.490 ± 0.073 | 4.217 ± 0.583 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.177 ± 0.009 | 0.464 ± 0.022 | 4.729 ± 1.273 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.181 ± 0.032 | 0.469 ± 0.020 | 4.915 ± 2.557 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.183 ± 0.013 | 0.424 ± 0.043 | 2.845 ± 0.779 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.189 ± 0.027 | 0.403 ± 0.024 | 2.319 ± 0.110 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.184 ± 0.010 | 0.417 ± 0.027 | 2.374 ± 0.142 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.330 ± 0.017 | 0.481 ± 0.059 | 4.845 ± 2.644 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.344 ± 0.028 | 0.461 ± 0.031 | 4.150 ± 1.157 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.337 ± 0.010 | 0.445 ± 0.040 | 4.994 ± 2.709 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.046 ± 0.004 | 0.692 ± 0.011 | 2.184 ± 0.100 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.048 ± 0.004 | 0.722 ± 0.087 | 2.063 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.048 ± 0.013 | 0.696 ± 0.066 | 2.371 ± 0.051 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.080 ± 0.007 | 0.457 ± 0.012 | 3.584 ± 1.378 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.082 ± 0.007 | 0.457 ± 0.010 | 4.214 ± 1.276 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.078 ± 0.005 | 0.460 ± 0.030 | 4.292 ± 1.254 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.048 ± 0.007 | 0.554 ± 0.066 | 1.478 ± 0.067 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.055 ± 0.002 | 0.450 ± 0.052 | 1.795 ± 0.078 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.066 ± 0.003 | 0.465 ± 0.035 | 1.719 ± 0.085 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.079 ± 0.009 | 0.366 ± 0.022 | 4.053 ± 1.126 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.089 ± 0.004 | 0.371 ± 0.110 | 3.509 ± 1.124 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.099 ± 0.006 | 0.377 ± 0.036 | 4.193 ± 1.902 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

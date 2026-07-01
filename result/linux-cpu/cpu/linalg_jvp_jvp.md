# Linux CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Run Inputs

- Threads 1: timestamp `20260701_104904`, raw run `data/results/linux-cpu/cpu/einsum/20260701_104904`
- Threads 4: timestamp `20260701_105129`, raw run `data/results/linux-cpu/cpu/einsum/20260701_105129`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260701_104904`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260701_104904`.

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

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260701_104904/cpu_ops_t1_20260701_104904.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260701_104904/linalg_jvp_vjp_t1_20260701_104904.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 11.540 ± 0.656 | 11.715 ± 0.083 | 35.840 ± 0.706 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 64.360 ± 1.010 | 70.619 ± 2.357 | 114.384 ± 1.333 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 11.191 ± 0.787 | 9.813 ± 0.302 | 34.773 ± 0.627 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 63.482 ± 1.052 | 52.093 ± 0.788 | 115.010 ± 1.944 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 7.388 ± 0.284 | 8.965 ± 0.269 | 10.589 ± 1.768 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 38.994 ± 3.477 | 66.622 ± 1.759 | 47.875 ± 2.678 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 7.242 ± 0.943 | 5.192 ± 0.031 | 14.618 ± 1.883 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 38.070 ± 3.114 | 37.420 ± 0.489 | 62.702 ± 4.783 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 8.411 ± 0.692 | 7.622 ± 0.013 | 18.451 ± 1.632 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 48.104 ± 3.929 | 50.516 ± 0.690 | 79.590 ± 10.450 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.699 ± 0.289 | 8.116 ± 0.532 | 19.546 ± 0.948 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 48.138 ± 6.216 | 49.602 ± 0.644 | 102.111 ± 2.391 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.607 ± 0.111 | 1.691 ± 0.010 | 5.171 ± 0.093 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.797 ± 0.478 | 5.569 ± 0.033 | 15.306 ± 0.534 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.293 ± 0.160 | 2.051 ± 0.006 | 5.869 ± 0.391 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.808 ± 0.709 | 8.917 ± 0.306 | 16.361 ± 0.177 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 22.150 ± 2.071 | 21.621 ± 0.144 | 61.573 ± 1.456 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 123.670 ± 2.950 | 140.210 ± 2.274 | 206.020 ± 5.916 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 23.295 ± 2.096 | 18.052 ± 0.111 | 60.531 ± 1.458 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 137.224 ± 3.549 | 117.573 ± 10.459 | 201.162 ± 4.499 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.001 | 0.337 ± 0.036 | 1.348 ± 0.041 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.043 ± 0.006 | 0.339 ± 0.011 | 1.368 ± 0.041 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.049 ± 0.004 | 0.385 ± 0.045 | 1.551 ± 0.059 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.037 ± 0.002 | 0.315 ± 0.014 | 3.142 ± 0.256 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.046 ± 0.007 | 0.326 ± 0.010 | 3.137 ± 0.298 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.068 ± 0.023 | 0.358 ± 0.053 | 4.018 ± 1.140 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.063 ± 0.002 | 0.457 ± 0.012 | 1.883 ± 0.094 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.066 ± 0.010 | 0.455 ± 0.013 | 1.963 ± 0.052 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.068 ± 0.011 | 0.463 ± 0.010 | 3.038 ± 0.165 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.069 ± 0.013 | 0.404 ± 0.011 | 4.461 ± 0.949 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.089 ± 0.025 | 0.416 ± 0.018 | 4.539 ± 1.021 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.085 ± 0.042 | 0.413 ± 0.009 | 4.944 ± 2.686 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.093 ± 0.011 | 0.361 ± 0.013 | 2.827 ± 0.181 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.109 ± 0.028 | 0.373 ± 0.069 | 2.287 ± 0.086 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.098 ± 0.002 | 0.384 ± 0.048 | 2.158 ± 0.082 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.068 ± 0.002 | 0.396 ± 0.015 | 4.778 ± 2.740 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.077 ± 0.018 | 0.399 ± 0.061 | 4.724 ± 1.663 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.086 ± 0.023 | 0.437 ± 0.028 | 4.889 ± 2.649 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.024 ± 0.003 | 0.715 ± 0.059 | 2.000 ± 0.056 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.023 ± 0.001 | 0.692 ± 0.012 | 1.667 ± 0.608 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.025 ± 0.005 | 0.700 ± 0.049 | 2.175 ± 0.027 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.032 ± 0.001 | 0.446 ± 0.011 | 4.178 ± 1.347 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.033 ± 0.010 | 0.445 ± 0.008 | 3.972 ± 0.892 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.036 ± 0.003 | 0.452 ± 0.012 | 4.244 ± 1.087 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.028 ± 0.001 | 0.436 ± 0.011 | 1.682 ± 0.070 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.049 ± 0.019 | 0.444 ± 0.012 | 1.705 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.050 ± 0.007 | 0.469 ± 0.038 | 1.665 ± 0.043 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.049 ± 0.001 | 0.349 ± 0.018 | 4.096 ± 1.629 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.056 ± 0.002 | 0.349 ± 0.011 | 3.895 ± 1.244 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.071 ± 0.012 | 0.367 ± 0.010 | 4.113 ± 1.629 |

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
- Timestamp: `20260701_105129`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260701_105129`.

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

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260701_105129/cpu_ops_t4_20260701_105129.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260701_105129/linalg_jvp_vjp_t4_20260701_105129.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.177 ± 0.255 | 7.590 ± 0.504 | 29.480 ± 0.340 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 32.594 ± 1.106 | 28.258 ± 0.639 | 88.522 ± 1.466 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 7.677 ± 0.234 | 6.670 ± 0.215 | 28.524 ± 0.263 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.233 ± 0.398 | 22.793 ± 0.560 | 88.097 ± 2.292 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 8.074 ± 1.800 | 5.680 ± 0.398 | 8.426 ± 2.025 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 33.168 ± 1.874 | 25.054 ± 1.046 | 24.612 ± 3.733 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 9.830 ± 1.581 | 2.854 ± 0.091 | 10.562 ± 0.664 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 34.917 ± 3.052 | 12.596 ± 0.981 | 32.890 ± 7.157 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.033 ± 0.310 | 4.743 ± 0.109 | 13.984 ± 0.857 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 28.836 ± 3.414 | 19.219 ± 0.410 | 45.313 ± 0.914 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 6.232 ± 0.176 | 4.511 ± 0.180 | 13.961 ± 2.577 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 28.495 ± 1.167 | 19.020 ± 0.534 | 57.407 ± 3.267 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.017 ± 0.026 | 1.636 ± 0.027 | 4.149 ± 0.085 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 7.681 ± 0.141 | 3.602 ± 0.129 | 10.357 ± 0.141 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.473 ± 0.092 | 1.735 ± 0.046 | 5.434 ± 0.427 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.962 ± 0.350 | 5.405 ± 0.114 | 10.792 ± 0.209 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.152 ± 2.091 | 18.558 ± 0.468 | 52.372 ± 1.387 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 72.298 ± 0.987 | 82.619 ± 6.397 | 162.552 ± 3.304 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 17.867 ± 0.408 | 16.461 ± 0.458 | 50.194 ± 2.285 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 78.410 ± 4.851 | 65.760 ± 2.359 | 162.860 ± 4.049 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.059 ± 0.010 | 0.324 ± 0.016 | 1.331 ± 0.063 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.074 ± 0.011 | 0.373 ± 0.096 | 1.393 ± 0.062 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.087 ± 0.019 | 0.350 ± 0.077 | 1.372 ± 0.054 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.064 ± 0.008 | 0.315 ± 0.009 | 3.089 ± 0.153 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.075 ± 0.014 | 0.348 ± 0.030 | 3.065 ± 0.102 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.088 ± 0.016 | 0.368 ± 0.063 | 4.032 ± 0.786 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.118 ± 0.009 | 0.558 ± 0.055 | 1.851 ± 0.074 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.115 ± 0.007 | 0.561 ± 0.038 | 2.034 ± 0.064 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.124 ± 0.008 | 0.541 ± 0.077 | 3.018 ± 0.411 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.110 ± 0.014 | 0.465 ± 0.014 | 4.142 ± 1.243 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.110 ± 0.006 | 0.480 ± 0.015 | 4.170 ± 1.229 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.117 ± 0.010 | 0.501 ± 0.019 | 4.633 ± 2.157 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.173 ± 0.028 | 0.416 ± 0.058 | 2.832 ± 0.270 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.179 ± 0.019 | 0.402 ± 0.031 | 2.440 ± 0.137 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.188 ± 0.024 | 0.415 ± 0.035 | 2.682 ± 0.152 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.116 ± 0.014 | 0.439 ± 0.042 | 4.770 ± 2.664 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.116 ± 0.007 | 0.439 ± 0.014 | 4.641 ± 2.019 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.122 ± 0.018 | 0.448 ± 0.068 | 4.909 ± 2.493 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.049 ± 0.009 | 0.689 ± 0.009 | 2.134 ± 0.090 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.046 ± 0.003 | 0.737 ± 0.131 | 2.134 ± 0.095 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.049 ± 0.004 | 0.744 ± 0.082 | 2.143 ± 0.056 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.077 ± 0.004 | 0.481 ± 0.065 | 3.459 ± 1.418 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.082 ± 0.005 | 0.455 ± 0.023 | 3.894 ± 1.227 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.081 ± 0.012 | 0.451 ± 0.009 | 3.630 ± 1.342 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.049 ± 0.001 | 0.434 ± 0.012 | 1.583 ± 0.079 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.060 ± 0.004 | 0.441 ± 0.027 | 1.786 ± 0.056 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.077 ± 0.004 | 0.462 ± 0.087 | 1.549 ± 0.044 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.081 ± 0.010 | 0.361 ± 0.036 | 3.875 ± 1.311 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.090 ± 0.008 | 0.353 ± 0.011 | 3.944 ± 1.115 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.018 | 0.374 ± 0.016 | 4.061 ± 1.683 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

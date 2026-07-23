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

- Threads 1: timestamp `20260723_051159`, raw run `data/results/linux-cpu/cpu/einsum/20260723_051159`
- Threads 4: timestamp `20260723_051422`, raw run `data/results/linux-cpu/cpu/einsum/20260723_051422`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260723_051159`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260723_051159`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_051159/cpu_ops_t1_20260723_051159.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_051159/linalg_jvp_vjp_t1_20260723_051159.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 13.602 ± 0.633 | 12.555 ± 0.601 | 25.778 ± 3.357 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 71.603 ± 3.051 | 78.399 ± 4.093 | 112.583 ± 1.720 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.807 ± 0.264 | 10.745 ± 0.535 | 25.491 ± 4.095 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 69.739 ± 1.521 | 63.293 ± 5.310 | 113.116 ± 0.892 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.203 ± 0.406 | 9.706 ± 0.548 | 13.372 ± 3.513 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 39.192 ± 5.747 | 88.148 ± 2.163 | 42.804 ± 3.429 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 7.950 ± 0.530 | 5.670 ± 0.387 | 13.666 ± 2.057 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 42.354 ± 4.043 | 48.487 ± 0.786 | 59.036 ± 4.300 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 10.678 ± 0.709 | 8.354 ± 0.374 | 16.928 ± 2.164 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 57.622 ± 6.154 | 56.472 ± 2.341 | 66.521 ± 10.492 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.278 ± 0.591 | 8.271 ± 0.338 | 18.244 ± 0.766 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 59.551 ± 2.553 | 54.328 ± 2.791 | 96.154 ± 7.511 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.521 ± 0.072 | 1.874 ± 0.242 | 4.352 ± 0.225 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.131 ± 0.484 | 6.792 ± 0.240 | 15.324 ± 0.432 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.175 ± 0.101 | 2.182 ± 0.247 | 4.377 ± 0.686 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.546 ± 0.306 | 11.062 ± 0.112 | 15.906 ± 0.544 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 20.677 ± 0.946 | 23.778 ± 0.801 | 48.500 ± 5.918 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 127.300 ± 2.963 | 154.625 ± 4.330 | 195.720 ± 6.562 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 19.893 ± 0.877 | 19.460 ± 0.639 | 56.982 ± 1.916 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 125.737 ± 3.269 | 123.852 ± 2.356 | 195.853 ± 2.194 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.110 ± 0.015 | 0.356 ± 0.100 | 1.543 ± 0.062 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.102 ± 0.020 | 0.378 ± 0.049 | 1.586 ± 0.033 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.120 ± 0.005 | 0.360 ± 0.031 | 1.577 ± 0.051 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.103 ± 0.016 | 0.332 ± 0.030 | 3.872 ± 0.202 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.109 ± 0.029 | 0.349 ± 0.039 | 3.469 ± 0.046 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.132 ± 0.015 | 0.358 ± 0.026 | 2.208 ± 0.123 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.095 ± 0.007 | 0.495 ± 0.083 | 1.996 ± 0.165 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.094 ± 0.021 | 0.485 ± 0.079 | 2.255 ± 0.024 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.105 ± 0.026 | 0.486 ± 0.022 | 1.480 ± 0.022 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.103 ± 0.015 | 0.441 ± 0.040 | 5.354 ± 0.165 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.108 ± 0.018 | 0.435 ± 0.033 | 6.005 ± 0.112 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.102 ± 0.027 | 0.451 ± 0.028 | 4.367 ± 0.794 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.230 ± 0.031 | 0.387 ± 0.030 | 2.164 ± 0.141 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.246 ± 0.039 | 0.406 ± 0.037 | 2.398 ± 0.049 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.242 ± 0.051 | 0.401 ± 0.079 | 2.127 ± 0.051 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.516 ± 0.058 | 0.419 ± 0.031 | 5.455 ± 0.280 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.530 ± 0.038 | 0.433 ± 0.018 | 5.471 ± 0.154 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.498 ± 0.054 | 0.434 ± 0.043 | 5.637 ± 0.189 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.060 ± 0.003 | 0.793 ± 0.102 | 1.737 ± 0.040 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.060 ± 0.016 | 0.730 ± 0.100 | 1.892 ± 0.038 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.070 ± 0.015 | 0.763 ± 0.032 | 1.715 ± 0.048 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.085 ± 0.010 | 0.471 ± 0.018 | 3.811 ± 0.230 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.086 ± 0.008 | 0.481 ± 0.030 | 4.198 ± 0.099 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.089 ± 0.019 | 0.548 ± 0.063 | 3.934 ± 0.776 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.058 ± 0.003 | 0.477 ± 0.063 | 1.457 ± 0.053 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.067 ± 0.004 | 0.468 ± 0.058 | 1.661 ± 0.026 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.081 ± 0.009 | 0.483 ± 0.020 | 1.582 ± 0.047 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.070 ± 0.012 | 0.376 ± 0.028 | 3.818 ± 0.173 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.080 ± 0.017 | 0.396 ± 0.041 | 4.360 ± 0.080 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.093 ± 0.016 | 0.435 ± 0.068 | 4.124 ± 0.179 |

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
- Timestamp: `20260723_051422`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260723_051422`.

- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_051422/cpu_ops_t4_20260723_051422.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_051422/linalg_jvp_vjp_t4_20260723_051422.md`

#### Linalg JVP/VJP Benchmark Items

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

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# Linux CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260723_035222 4:20260723_042457 permutation:20260723_051703`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260723_035222`, raw run `data/results/linux-cpu/cpu/einsum/20260723_035222`
- Threads 4: timestamp `20260723_042457`, raw run `data/results/linux-cpu/cpu/einsum/20260723_042457`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260723_035222`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260723_035222`.

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_035222/cpu_ops_t1_20260723_035222.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_035222/linalg_jvp_vjp_t1_20260723_035222.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 12.684 ± 0.622 | 12.219 ± 0.245 | 33.582 ± 0.244 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 70.333 ± 0.486 | 80.834 ± 1.245 | 112.353 ± 0.657 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.562 ± 0.339 | 10.203 ± 0.259 | 32.987 ± 0.269 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 68.780 ± 0.747 | 59.669 ± 1.486 | 114.540 ± 1.363 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 7.788 ± 0.297 | 9.464 ± 0.261 | 11.413 ± 1.823 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 43.059 ± 2.141 | 72.511 ± 0.900 | 43.025 ± 3.329 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 7.168 ± 0.523 | 5.460 ± 0.025 | 14.104 ± 0.907 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 42.527 ± 0.279 | 40.201 ± 2.188 | 51.466 ± 1.233 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 10.369 ± 0.314 | 8.054 ± 0.112 | 17.175 ± 1.551 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 57.296 ± 2.150 | 52.903 ± 0.487 | 75.656 ± 2.339 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 10.792 ± 0.602 | 7.689 ± 0.233 | 18.757 ± 1.389 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 57.963 ± 0.628 | 51.758 ± 1.590 | 97.342 ± 4.293 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.512 ± 0.044 | 1.780 ± 0.016 | 4.591 ± 0.212 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.629 ± 0.185 | 5.794 ± 0.318 | 14.915 ± 0.282 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.223 ± 0.081 | 2.158 ± 0.021 | 4.359 ± 0.067 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.975 ± 0.102 | 9.391 ± 0.268 | 15.763 ± 0.478 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 20.292 ± 0.128 | 23.061 ± 0.210 | 58.292 ± 2.898 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 123.906 ± 0.528 | 147.386 ± 1.131 | 217.923 ± 27.369 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 19.885 ± 0.147 | 19.124 ± 0.258 | 56.446 ± 16.246 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 122.615 ± 1.553 | 119.296 ± 1.170 | 178.635 ± 21.268 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.090 ± 0.001 | 0.360 ± 0.041 | 1.242 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.099 ± 0.011 | 0.355 ± 0.028 | 1.255 ± 0.013 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.105 ± 0.009 | 0.356 ± 0.035 | 1.632 ± 0.024 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.100 ± 0.008 | 0.330 ± 0.032 | 3.536 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.107 ± 0.001 | 0.345 ± 0.029 | 3.562 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.114 ± 0.004 | 0.359 ± 0.031 | 4.148 ± 0.035 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.094 ± 0.011 | 0.464 ± 0.042 | 1.710 ± 0.042 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.091 ± 0.013 | 0.468 ± 0.015 | 1.794 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.093 ± 0.007 | 0.484 ± 0.016 | 2.563 ± 0.026 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.097 ± 0.011 | 0.433 ± 0.023 | 5.209 ± 0.045 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.105 ± 0.021 | 0.429 ± 0.045 | 5.651 ± 0.515 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.106 ± 0.019 | 0.439 ± 0.028 | 6.515 ± 1.208 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.209 ± 0.018 | 0.373 ± 0.035 | 2.677 ± 0.033 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.221 ± 0.027 | 0.381 ± 0.022 | 2.363 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.228 ± 0.016 | 0.376 ± 0.037 | 2.548 ± 0.040 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.506 ± 0.027 | 0.405 ± 0.032 | 6.064 ± 0.995 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.502 ± 0.046 | 0.413 ± 0.035 | 6.049 ± 0.741 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.490 ± 0.006 | 0.429 ± 0.053 | 6.190 ± 0.981 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.088 ± 0.029 | 0.714 ± 0.029 | 2.098 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.058 ± 0.003 | 0.713 ± 0.022 | 2.242 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.059 ± 0.008 | 0.724 ± 0.068 | 2.061 ± 0.027 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.083 ± 0.012 | 0.465 ± 0.016 | 4.430 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.082 ± 0.002 | 0.473 ± 0.083 | 4.450 ± 0.048 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.085 ± 0.003 | 0.475 ± 0.028 | 4.627 ± 0.069 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.057 ± 0.005 | 0.452 ± 0.011 | 1.780 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.063 ± 0.002 | 0.456 ± 0.030 | 1.961 ± 0.019 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.079 ± 0.004 | 0.482 ± 0.045 | 1.660 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.067 ± 0.001 | 0.396 ± 0.116 | 4.325 ± 0.069 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.075 ± 0.004 | 0.372 ± 0.035 | 4.395 ± 0.108 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.090 ± 0.008 | 0.413 ± 0.095 | 4.503 ± 0.143 |

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
- Timestamp: `20260723_042457`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260723_042457`.

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260723_042457/cpu_ops_t4_20260723_042457.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260723_042457/linalg_jvp_vjp_t4_20260723_042457.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.173 ± 0.073 | 7.977 ± 0.397 | 30.567 ± 0.522 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 30.303 ± 0.384 | 28.180 ± 0.272 | 88.076 ± 2.704 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 7.758 ± 0.259 | 6.779 ± 0.129 | 29.475 ± 0.470 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.052 ± 0.862 | 23.267 ± 0.574 | 89.556 ± 3.272 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.701 ± 0.527 | 4.548 ± 0.144 | 8.193 ± 1.669 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 24.414 ± 0.485 | 24.724 ± 1.446 | 25.470 ± 0.793 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 4.877 ± 0.553 | 2.962 ± 0.170 | 10.804 ± 1.812 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 22.778 ± 0.554 | 12.157 ± 0.396 | 35.680 ± 6.376 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.520 ± 0.588 | 4.514 ± 0.063 | 14.114 ± 0.886 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 29.933 ± 1.123 | 19.717 ± 0.543 | 44.192 ± 1.527 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 7.130 ± 0.979 | 4.384 ± 0.154 | 14.752 ± 2.000 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 28.096 ± 0.366 | 19.489 ± 0.918 | 61.908 ± 2.113 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.450 ± 0.062 | 1.664 ± 0.031 | 4.117 ± 0.073 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.517 ± 0.283 | 3.288 ± 0.217 | 10.569 ± 0.736 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.897 ± 0.024 | 1.773 ± 0.023 | 5.198 ± 0.149 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.525 ± 0.203 | 4.664 ± 0.224 | 11.329 ± 0.821 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.542 ± 0.224 | 21.703 ± 3.839 | 53.130 ± 0.596 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 72.295 ± 0.196 | 82.333 ± 6.689 | 162.393 ± 2.778 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.955 ± 0.036 | 16.514 ± 0.439 | 51.212 ± 1.665 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 70.770 ± 1.352 | 65.032 ± 1.764 | 163.880 ± 4.562 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.114 ± 0.011 | 0.339 ± 0.028 | 1.470 ± 0.033 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.124 ± 0.029 | 0.353 ± 0.011 | 1.201 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.128 ± 0.022 | 0.408 ± 0.018 | 1.721 ± 0.021 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.123 ± 0.031 | 0.320 ± 0.009 | 3.827 ± 0.155 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.130 ± 0.013 | 0.327 ± 0.013 | 3.503 ± 0.060 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.134 ± 0.028 | 0.343 ± 0.035 | 4.578 ± 0.042 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.123 ± 0.020 | 0.522 ± 0.012 | 1.906 ± 0.051 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.128 ± 0.008 | 0.508 ± 0.032 | 1.837 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.119 ± 0.006 | 0.524 ± 0.013 | 2.832 ± 0.102 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.126 ± 0.022 | 0.477 ± 0.014 | 5.502 ± 0.124 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.121 ± 0.024 | 0.478 ± 0.025 | 5.966 ± 0.564 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.131 ± 0.013 | 0.474 ± 0.077 | 6.469 ± 1.336 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.365 ± 0.104 | 0.408 ± 0.019 | 3.181 ± 0.028 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.295 ± 0.053 | 0.410 ± 0.031 | 2.966 ± 0.032 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.307 ± 0.046 | 0.425 ± 0.052 | 2.812 ± 0.036 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.644 ± 0.050 | 0.483 ± 0.042 | 7.013 ± 1.481 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.612 ± 0.016 | 0.438 ± 0.033 | 6.283 ± 0.939 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.576 ± 0.085 | 0.439 ± 0.060 | 6.738 ± 1.232 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.081 ± 0.017 | 0.689 ± 0.006 | 2.289 ± 0.033 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.076 ± 0.017 | 0.703 ± 0.104 | 2.465 ± 0.078 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.076 ± 0.011 | 0.758 ± 0.160 | 2.311 ± 0.037 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.103 ± 0.003 | 0.451 ± 0.052 | 4.618 ± 0.486 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.105 ± 0.011 | 0.457 ± 0.011 | 5.062 ± 0.181 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.104 ± 0.013 | 0.462 ± 0.054 | 4.527 ± 0.045 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.074 ± 0.002 | 0.431 ± 0.010 | 1.899 ± 0.078 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.081 ± 0.008 | 0.470 ± 0.021 | 2.062 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.092 ± 0.004 | 0.587 ± 0.087 | 1.791 ± 0.049 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.080 ± 0.008 | 0.393 ± 0.065 | 5.046 ± 0.317 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.094 ± 0.018 | 0.365 ± 0.032 | 4.968 ± 0.391 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.013 | 0.389 ± 0.054 | 4.995 ± 0.270 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

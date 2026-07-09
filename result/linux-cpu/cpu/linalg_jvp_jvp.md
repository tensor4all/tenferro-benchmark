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

- Threads 1: timestamp `20260709_160834`, raw run `data/results/linux-cpu/cpu/einsum/20260709_160834`
- Threads 4: timestamp `20260709_161107`, raw run `data/results/linux-cpu/cpu/einsum/20260709_161107`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260709_160834`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260709_160834`.

- tenferro-rs commit: `b17416e8ddce34876c73a583178ba83c8c8dc8a9`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260709_160834/cpu_ops_t1_20260709_160834.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260709_160834/linalg_jvp_vjp_t1_20260709_160834.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 13.650 ± 0.333 | 14.815 ± 0.347 | 32.486 ± 9.812 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 81.572 ± 0.847 | 87.850 ± 0.372 | 105.131 ± 6.339 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 13.070 ± 0.327 | 12.324 ± 0.297 | 33.417 ± 0.673 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 79.052 ± 0.382 | 64.473 ± 0.537 | 109.763 ± 3.378 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 7.538 ± 0.348 | 11.701 ± 0.329 | 10.056 ± 1.604 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 43.539 ± 4.218 | 88.948 ± 0.896 | 46.919 ± 2.526 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 6.988 ± 0.149 | 6.785 ± 0.118 | 15.225 ± 1.075 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 43.430 ± 4.253 | 49.446 ± 0.595 | 57.016 ± 7.178 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.323 ± 0.293 | 9.316 ± 0.207 | 15.732 ± 2.266 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 58.444 ± 5.708 | 68.562 ± 1.979 | 61.699 ± 14.928 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 9.409 ± 0.380 | 9.041 ± 0.364 | 17.348 ± 1.955 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 60.227 ± 1.418 | 64.695 ± 1.536 | 73.651 ± 7.213 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.609 ± 0.076 | 2.093 ± 0.089 | 4.679 ± 0.150 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 9.195 ± 0.121 | 7.478 ± 0.258 | 15.429 ± 0.485 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.059 ± 0.041 | 2.485 ± 0.019 | 5.033 ± 0.461 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.892 ± 0.234 | 11.799 ± 0.526 | 15.821 ± 0.402 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 23.138 ± 1.057 | 26.626 ± 0.525 | 58.170 ± 1.292 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 142.106 ± 3.147 | 177.697 ± 1.094 | 195.795 ± 4.246 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 22.465 ± 0.248 | 22.601 ± 0.329 | 55.373 ± 3.571 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 140.204 ± 1.829 | 142.134 ± 1.693 | 198.945 ± 15.931 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.044 ± 0.000 | 0.399 ± 0.021 | 1.051 ± 0.020 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.053 ± 0.003 | 0.394 ± 0.017 | 1.053 ± 0.019 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.061 ± 0.007 | 0.410 ± 0.018 | 1.292 ± 0.033 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.050 ± 0.002 | 0.372 ± 0.013 | 3.106 ± 0.188 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.058 ± 0.002 | 0.390 ± 0.020 | 3.129 ± 0.196 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.070 ± 0.014 | 0.391 ± 0.017 | 3.856 ± 1.111 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.047 ± 0.002 | 0.547 ± 0.009 | 1.729 ± 0.035 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.050 ± 0.002 | 0.537 ± 0.016 | 1.660 ± 0.028 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.051 ± 0.003 | 0.531 ± 0.007 | 2.697 ± 0.032 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.044 ± 0.002 | 0.486 ± 0.012 | 3.924 ± 1.358 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.046 ± 0.001 | 0.481 ± 0.013 | 4.423 ± 1.430 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.047 ± 0.003 | 0.482 ± 0.012 | 4.313 ± 2.400 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.120 ± 0.005 | 0.428 ± 0.018 | 2.696 ± 0.188 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.124 ± 0.004 | 0.430 ± 0.024 | 2.234 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.132 ± 0.020 | 0.429 ± 0.023 | 2.212 ± 0.022 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.266 ± 0.008 | 0.473 ± 0.021 | 4.529 ± 2.471 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.266 ± 0.013 | 0.467 ± 0.031 | 4.402 ± 2.583 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.309 ± 0.018 | 0.465 ± 0.017 | 4.448 ± 2.537 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.027 ± 0.001 | 0.829 ± 0.011 | 1.944 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.029 ± 0.001 | 0.831 ± 0.016 | 1.902 ± 0.052 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.029 ± 0.001 | 0.839 ± 0.025 | 1.998 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.040 ± 0.001 | 0.533 ± 0.010 | 3.880 ± 1.589 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.038 ± 0.001 | 0.537 ± 0.014 | 3.994 ± 1.044 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.040 ± 0.001 | 0.541 ± 0.017 | 3.907 ± 1.061 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.036 ± 0.004 | 0.522 ± 0.014 | 1.573 ± 0.025 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.043 ± 0.001 | 0.512 ± 0.012 | 1.618 ± 0.019 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.062 ± 0.004 | 0.541 ± 0.025 | 1.531 ± 0.017 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.038 ± 0.001 | 0.408 ± 0.030 | 3.833 ± 1.592 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.047 ± 0.001 | 0.408 ± 0.016 | 3.741 ± 1.439 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.068 ± 0.013 | 0.434 ± 0.014 | 3.867 ± 1.447 |

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
- Timestamp: `20260709_161107`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260709_161107`.

- tenferro-rs commit: `b17416e8ddce34876c73a583178ba83c8c8dc8a9`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260709_161107/cpu_ops_t4_20260709_161107.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260709_161107/linalg_jvp_vjp_t4_20260709_161107.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.140 ± 0.214 | 9.635 ± 2.308 | 29.867 ± 0.593 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 30.035 ± 0.553 | 30.578 ± 0.291 | 86.497 ± 8.999 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 7.951 ± 0.252 | 6.671 ± 0.201 | 29.082 ± 1.137 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 28.714 ± 0.425 | 22.636 ± 0.480 | 88.075 ± 4.050 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.553 ± 0.379 | 4.369 ± 0.095 | 7.159 ± 1.811 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.174 ± 5.049 | 27.124 ± 0.235 | 24.536 ± 2.964 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.154 ± 0.327 | 2.806 ± 0.049 | 10.474 ± 0.584 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 19.230 ± 3.728 | 12.887 ± 0.396 | 29.096 ± 5.480 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 6.968 ± 0.275 | 4.641 ± 0.047 | 13.689 ± 0.858 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 28.491 ± 1.638 | 20.895 ± 0.299 | 41.560 ± 10.268 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 7.035 ± 0.549 | 4.481 ± 0.073 | 13.737 ± 0.331 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 27.391 ± 1.064 | 20.828 ± 0.300 | 41.876 ± 5.616 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.455 ± 0.032 | 1.594 ± 0.018 | 4.153 ± 0.116 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.413 ± 0.135 | 3.391 ± 0.100 | 10.367 ± 0.085 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.931 ± 0.042 | 1.685 ± 0.005 | 5.225 ± 0.373 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.250 ± 0.130 | 5.003 ± 0.038 | 10.874 ± 0.150 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.125 ± 2.759 | 19.532 ± 0.476 | 51.620 ± 1.678 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 73.867 ± 1.551 | 85.563 ± 3.659 | 158.758 ± 9.588 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.586 ± 0.222 | 17.210 ± 0.263 | 50.943 ± 2.098 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 73.838 ± 2.124 | 70.251 ± 0.615 | 154.489 ± 6.287 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.062 ± 0.014 | 0.395 ± 0.077 | 1.068 ± 0.047 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.073 ± 0.006 | 0.421 ± 0.076 | 1.085 ± 0.034 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.081 ± 0.005 | 0.386 ± 0.060 | 1.249 ± 0.019 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.067 ± 0.009 | 0.358 ± 0.040 | 2.814 ± 0.409 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.080 ± 0.011 | 0.389 ± 0.050 | 2.650 ± 0.774 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.088 ± 0.003 | 0.372 ± 0.026 | 3.699 ± 1.130 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.076 ± 0.004 | 0.707 ± 0.040 | 1.640 ± 0.040 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.078 ± 0.006 | 0.713 ± 0.046 | 1.611 ± 0.053 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.084 ± 0.010 | 0.719 ± 0.044 | 2.250 ± 0.040 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.078 ± 0.012 | 0.610 ± 0.041 | 3.875 ± 1.310 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.078 ± 0.005 | 0.603 ± 0.032 | 3.894 ± 1.374 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.081 ± 0.009 | 0.608 ± 0.042 | 4.581 ± 2.584 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.172 ± 0.012 | 0.586 ± 0.044 | 2.524 ± 0.034 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.176 ± 0.006 | 0.564 ± 0.038 | 2.134 ± 0.054 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.185 ± 0.018 | 0.561 ± 0.069 | 2.135 ± 0.041 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.425 ± 0.032 | 0.594 ± 0.039 | 3.535 ± 0.738 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.395 ± 0.017 | 0.606 ± 0.051 | 4.468 ± 1.770 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.412 ± 0.034 | 0.570 ± 0.049 | 3.690 ± 1.554 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.045 ± 0.003 | 0.758 ± 0.044 | 1.962 ± 0.018 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.047 ± 0.009 | 0.752 ± 0.068 | 1.796 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.048 ± 0.006 | 0.749 ± 0.026 | 2.165 ± 0.028 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.062 ± 0.011 | 0.489 ± 0.035 | 3.602 ± 1.215 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.063 ± 0.005 | 0.504 ± 0.045 | 3.976 ± 1.061 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.067 ± 0.007 | 0.492 ± 0.062 | 3.980 ± 1.153 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.048 ± 0.004 | 0.446 ± 0.013 | 1.561 ± 0.077 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.058 ± 0.002 | 0.541 ± 0.055 | 1.149 ± 0.448 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.077 ± 0.002 | 0.513 ± 0.045 | 1.475 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.052 ± 0.001 | 0.414 ± 0.074 | 3.828 ± 1.378 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.063 ± 0.002 | 0.413 ± 0.039 | 2.760 ± 0.765 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.081 ± 0.001 | 0.401 ± 0.014 | 3.218 ± 0.573 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

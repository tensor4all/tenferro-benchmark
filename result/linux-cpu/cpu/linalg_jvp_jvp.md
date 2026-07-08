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

- Threads 1: timestamp `20260708_084134`, raw run `data/results/linux-cpu/cpu/einsum/20260708_084134`
- Threads 4: timestamp `20260708_084507`, raw run `data/results/linux-cpu/cpu/einsum/20260708_084507`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260708_084134`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260708_084134`.

- tenferro-rs commit: `bcdb0d5aa3d181335fde627d359053ae218d8250`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260708_084134/cpu_ops_t1_20260708_084134.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260708_084134/linalg_jvp_vjp_t1_20260708_084134.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 12.285 ± 0.485 | 12.640 ± 0.506 | 25.986 ± 3.631 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 68.804 ± 1.790 | 138.267 ± 1.105 | 122.186 ± 8.673 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.089 ± 0.534 | 10.775 ± 0.513 | 27.454 ± 3.207 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 68.256 ± 9.795 | 98.790 ± 2.126 | 119.508 ± 5.651 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.123 ± 0.493 | 9.537 ± 0.240 | 13.206 ± 3.825 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 45.052 ± 10.144 | 160.299 ± 7.706 | 51.332 ± 9.847 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 7.558 ± 0.369 | 6.231 ± 0.830 | 15.668 ± 2.378 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 39.879 ± 3.168 | 93.219 ± 5.244 | 61.712 ± 7.292 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 8.936 ± 0.594 | 8.145 ± 0.547 | 20.132 ± 5.049 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 50.201 ± 4.725 | 55.027 ± 8.268 | 78.787 ± 11.086 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 8.607 ± 0.363 | 7.904 ± 0.513 | 21.651 ± 3.512 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 51.573 ± 2.464 | 105.562 ± 51.489 | 97.013 ± 11.355 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.426 ± 0.064 | 1.847 ± 0.232 | 4.342 ± 0.182 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.553 ± 0.232 | 14.074 ± 1.161 | 15.330 ± 1.865 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.012 ± 0.013 | 2.124 ± 0.393 | 5.233 ± 0.348 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.731 ± 0.415 | 18.742 ± 0.587 | 15.796 ± 2.736 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 21.678 ± 0.887 | 24.466 ± 1.369 | 44.761 ± 2.822 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 133.264 ± 10.049 | 144.681 ± 4.422 | 242.207 ± 14.292 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 21.050 ± 0.547 | 19.757 ± 0.763 | 47.149 ± 4.717 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 149.695 ± 26.247 | 118.790 ± 5.255 | 236.604 ± 12.417 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.041 ± 0.002 | 0.326 ± 0.014 | 1.507 ± 0.091 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.048 ± 0.001 | 0.387 ± 0.063 | 1.798 ± 0.082 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.054 ± 0.006 | 0.346 ± 0.011 | 1.389 ± 0.090 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.045 ± 0.004 | 0.318 ± 0.010 | 4.202 ± 0.262 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.053 ± 0.004 | 0.365 ± 0.082 | 4.383 ± 0.398 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.061 ± 0.019 | 0.336 ± 0.012 | 4.520 ± 0.378 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.059 ± 0.006 | 0.450 ± 0.011 | 2.140 ± 0.109 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.064 ± 0.004 | 0.524 ± 0.073 | 2.625 ± 0.106 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.075 ± 0.024 | 0.458 ± 0.011 | 2.403 ± 0.178 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.058 ± 0.004 | 0.412 ± 0.019 | 6.394 ± 0.315 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.058 ± 0.004 | 0.436 ± 0.042 | 6.856 ± 0.252 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.065 ± 0.010 | 0.445 ± 0.069 | 5.773 ± 3.124 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.112 ± 0.016 | 0.361 ± 0.012 | 2.110 ± 0.187 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.109 ± 0.009 | 0.381 ± 0.031 | 2.609 ± 0.120 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.117 ± 0.028 | 0.373 ± 0.012 | 2.343 ± 0.315 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.260 ± 0.011 | 0.391 ± 0.008 | 6.354 ± 0.855 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.262 ± 0.019 | 0.405 ± 0.048 | 6.104 ± 0.160 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.259 ± 0.014 | 0.407 ± 0.024 | 6.188 ± 1.087 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.024 ± 0.001 | 0.684 ± 0.034 | 1.778 ± 0.107 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.025 ± 0.002 | 0.750 ± 0.064 | 1.905 ± 0.098 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.026 ± 0.002 | 0.700 ± 0.046 | 2.346 ± 0.407 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.037 ± 0.011 | 0.451 ± 0.044 | 4.450 ± 0.221 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.035 ± 0.010 | 0.452 ± 0.042 | 4.613 ± 0.174 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.036 ± 0.001 | 0.470 ± 0.043 | 4.322 ± 0.438 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.032 ± 0.002 | 0.448 ± 0.056 | 1.447 ± 0.368 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.042 ± 0.002 | 0.438 ± 0.027 | 1.719 ± 0.143 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.060 ± 0.003 | 0.454 ± 0.009 | 1.516 ± 0.044 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.036 ± 0.001 | 0.361 ± 0.061 | 4.993 ± 0.285 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.045 ± 0.002 | 0.354 ± 0.037 | 4.579 ± 0.192 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.058 ± 0.004 | 0.380 ± 0.071 | 4.812 ± 0.179 |

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
- Timestamp: `20260708_084507`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260708_084507`.

- tenferro-rs commit: `bcdb0d5aa3d181335fde627d359053ae218d8250`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260708_084507/cpu_ops_t4_20260708_084507.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260708_084507/linalg_jvp_vjp_t4_20260708_084507.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.321 ± 2.521 | 7.652 ± 1.115 | 24.331 ± 2.885 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 46.388 ± 0.986 | 156.462 ± 287.295 | 92.305 ± 9.800 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.763 ± 0.134 | 6.490 ± 0.286 | 26.738 ± 9.092 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 43.707 ± 0.357 | 47.678 ± 21.398 | 85.280 ± 13.202 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 11.146 ± 2.698 | 4.357 ± 0.054 | 13.006 ± 6.030 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 37.383 ± 9.201 | 147.215 ± 138.796 | 38.526 ± 4.196 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 11.135 ± 3.197 | 2.760 ± 0.121 | 15.329 ± 4.259 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 38.151 ± 6.159 | 41.110 ± 35.361 | 39.251 ± 8.679 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 10.388 ± 3.347 | 4.652 ± 0.204 | 17.242 ± 5.812 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 42.476 ± 7.257 | 78.230 ± 55.280 | 58.257 ± 6.139 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 10.149 ± 2.321 | 4.388 ± 0.044 | 17.740 ± 3.128 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 42.456 ± 0.997 | 35.013 ± 0.646 | 72.687 ± 11.953 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.167 ± 0.060 | 1.660 ± 0.020 | 5.000 ± 1.772 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 8.721 ± 0.400 | 9.160 ± 19.021 | 10.736 ± 3.211 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.208 ± 0.109 | 1.767 ± 0.046 | 6.456 ± 3.512 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 5.571 ± 0.122 | 10.926 ± 0.804 | 14.957 ± 3.168 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 24.939 ± 0.976 | 18.826 ± 0.773 | 42.736 ± 2.844 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 105.138 ± 1.396 | 82.814 ± 7.444 | 191.187 ± 13.194 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 23.936 ± 0.151 | 17.401 ± 0.748 | 40.413 ± 6.343 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 105.193 ± 2.591 | 77.451 ± 190.436 | 185.773 ± 12.072 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.169 ± 0.010 | 0.400 ± 0.015 | 1.483 ± 0.043 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.191 ± 0.009 | 0.409 ± 0.011 | 1.684 ± 0.038 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.205 ± 0.019 | 0.420 ± 0.014 | 1.518 ± 0.088 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.190 ± 0.028 | 0.379 ± 0.012 | 4.336 ± 0.650 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.217 ± 0.049 | 0.390 ± 0.013 | 4.644 ± 0.314 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.246 ± 0.044 | 0.405 ± 0.012 | 4.551 ± 0.400 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.315 ± 0.013 | 0.585 ± 0.015 | 2.445 ± 0.434 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.326 ± 0.073 | 0.581 ± 0.014 | 2.719 ± 0.198 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.326 ± 0.022 | 0.587 ± 0.014 | 2.567 ± 0.442 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.279 ± 0.021 | 0.514 ± 0.012 | 6.560 ± 0.857 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.298 ± 0.050 | 0.511 ± 0.014 | 5.963 ± 0.602 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.299 ± 0.025 | 0.519 ± 0.011 | 6.733 ± 0.438 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.440 ± 0.607 | 0.454 ± 0.012 | 2.156 ± 0.123 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.402 ± 0.053 | 0.456 ± 0.012 | 2.810 ± 0.471 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.478 ± 0.161 | 0.462 ± 0.011 | 2.331 ± 0.138 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.739 ± 0.528 | 0.491 ± 0.011 | 5.336 ± 0.198 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.960 ± 1.406 | 0.490 ± 0.012 | 6.646 ± 0.820 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.847 ± 0.092 | 0.499 ± 0.012 | 6.427 ± 0.341 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.133 ± 0.006 | 0.834 ± 0.010 | 1.876 ± 0.164 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.166 ± 0.114 | 0.831 ± 0.009 | 1.908 ± 0.395 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.135 ± 0.006 | 0.834 ± 0.005 | 1.806 ± 0.130 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.188 ± 0.012 | 0.536 ± 0.011 | 5.055 ± 0.724 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.191 ± 0.007 | 0.543 ± 0.011 | 5.201 ± 0.689 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.158 ± 0.031 | 0.553 ± 0.015 | 4.784 ± 0.296 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.125 ± 0.005 | 0.846 ± 0.029 | 1.857 ± 0.179 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.115 ± 0.004 | 0.543 ± 0.030 | 1.534 ± 0.072 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.187 ± 0.036 | 0.563 ± 0.013 | 1.612 ± 0.147 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.139 ± 0.008 | 0.420 ± 0.020 | 4.933 ± 0.773 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.143 ± 0.071 | 0.422 ± 0.014 | 5.306 ± 0.454 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.193 ± 0.010 | 0.443 ± 0.012 | 4.515 ± 0.527 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

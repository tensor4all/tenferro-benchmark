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

- Threads 1: timestamp `20260630_142012`, raw run `data/results/linux-cpu/cpu/einsum/20260630_142012`
- Threads 4: timestamp `20260630_142316`, raw run `data/results/linux-cpu/cpu/einsum/20260630_142316`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260630_142012`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_142012`.

- tenferro-rs commit: `061093d66ee9ff9dd9fa49d9b14f30ae4cc6e9a1`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_142012/cpu_ops_t1_20260630_142012.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_142012/linalg_jvp_vjp_t1_20260630_142012.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.105 ± 0.440 | 12.106 ± 0.468 | 35.500 ± 0.589 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 79.861 ± 1.447 | 71.650 ± 1.110 | 114.817 ± 2.003 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 14.511 ± 0.236 | 9.987 ± 0.452 | 35.061 ± 0.427 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 74.205 ± 3.455 | 53.104 ± 1.037 | 114.474 ± 1.561 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.622 ± 0.673 | 11.570 ± 1.040 | 10.627 ± 2.447 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 42.027 ± 5.539 | 68.243 ± 0.352 | 49.087 ± 1.554 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 9.108 ± 0.530 | 5.260 ± 0.342 | 15.813 ± 2.289 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 43.227 ± 3.315 | 38.356 ± 0.672 | 54.063 ± 1.371 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 10.578 ± 0.423 | 7.828 ± 0.463 | 18.246 ± 1.123 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 58.395 ± 13.067 | 51.528 ± 1.554 | 70.556 ± 10.173 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 9.928 ± 0.246 | 7.511 ± 0.298 | 19.231 ± 1.182 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 59.014 ± 1.395 | 50.932 ± 1.776 | 101.069 ± 3.878 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.994 ± 0.107 | 1.715 ± 0.221 | 5.077 ± 0.059 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.289 ± 0.551 | 5.711 ± 0.215 | 15.685 ± 0.334 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.575 ± 0.133 | 2.027 ± 0.209 | 5.683 ± 0.315 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 7.123 ± 0.215 | 11.247 ± 0.456 | 16.190 ± 0.133 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 25.224 ± 2.474 | 22.507 ± 1.353 | 60.910 ± 0.832 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 146.452 ± 2.397 | 144.202 ± 2.274 | 207.611 ± 3.660 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 27.875 ± 1.542 | 18.602 ± 0.435 | 59.786 ± 1.653 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 159.038 ± 3.785 | 116.370 ± 1.445 | 204.260 ± 4.260 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.044 ± 0.007 | 0.311 ± 0.010 | 1.344 ± 0.050 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.048 ± 0.001 | 0.334 ± 0.047 | 1.361 ± 0.053 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.043 ± 0.004 | 0.328 ± 0.012 | 1.527 ± 0.062 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.046 ± 0.003 | 0.311 ± 0.034 | 3.165 ± 0.435 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.053 ± 0.003 | 0.315 ± 0.015 | 3.117 ± 0.207 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.046 ± 0.001 | 0.316 ± 0.011 | 4.106 ± 1.361 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.078 ± 0.002 | 0.442 ± 0.010 | 1.927 ± 0.052 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.082 ± 0.014 | 0.447 ± 0.058 | 1.816 ± 0.094 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.064 ± 0.003 | 0.444 ± 0.035 | 2.975 ± 0.332 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.086 ± 0.003 | 0.393 ± 0.011 | 4.564 ± 1.416 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.088 ± 0.018 | 0.419 ± 0.049 | 4.910 ± 1.393 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.068 ± 0.006 | 0.401 ± 0.041 | 5.001 ± 2.636 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.125 ± 0.031 | 0.353 ± 0.046 | 3.018 ± 0.218 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.125 ± 0.002 | 0.341 ± 0.010 | 2.217 ± 0.130 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.096 ± 0.024 | 0.353 ± 0.062 | 2.331 ± 0.061 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.087 ± 0.003 | 0.377 ± 0.010 | 4.581 ± 2.794 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.094 ± 0.004 | 0.375 ± 0.014 | 4.661 ± 2.367 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.074 ± 0.007 | 0.377 ± 0.011 | 4.870 ± 2.736 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.028 ± 0.002 | 0.658 ± 0.027 | 2.208 ± 0.083 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.029 ± 0.002 | 0.664 ± 0.109 | 2.036 ± 0.093 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.001 | 0.662 ± 0.022 | 2.482 ± 0.025 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.039 ± 0.002 | 0.441 ± 0.050 | 4.092 ± 1.314 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.040 ± 0.002 | 0.431 ± 0.011 | 4.057 ± 1.123 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.030 ± 0.002 | 0.441 ± 0.029 | 4.187 ± 0.981 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.032 ± 0.002 | 0.421 ± 0.083 | 1.571 ± 0.076 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.038 ± 0.001 | 0.422 ± 0.062 | 1.668 ± 0.023 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.044 ± 0.004 | 0.439 ± 0.008 | 1.684 ± 0.068 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.057 ± 0.002 | 0.336 ± 0.017 | 3.844 ± 1.250 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.066 ± 0.002 | 0.338 ± 0.011 | 3.974 ± 1.079 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.062 ± 0.003 | 0.358 ± 0.027 | 4.069 ± 1.656 |

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
- Timestamp: `20260630_142316`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_142316`.

- tenferro-rs commit: `061093d66ee9ff9dd9fa49d9b14f30ae4cc6e9a1`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_142316/cpu_ops_t4_20260630_142316.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_142316/linalg_jvp_vjp_t4_20260630_142316.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 13.451 ± 0.280 | 7.868 ± 1.167 | 29.722 ± 0.692 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 56.639 ± 0.622 | 27.406 ± 4.999 | 87.846 ± 2.292 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 13.348 ± 0.405 | 6.201 ± 0.283 | 28.637 ± 0.335 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 54.955 ± 0.684 | 22.231 ± 1.652 | 87.909 ± 1.866 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 9.003 ± 1.241 | 4.038 ± 0.199 | 6.806 ± 1.832 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 35.841 ± 2.469 | 24.511 ± 1.640 | 24.252 ± 3.604 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 10.247 ± 1.616 | 2.617 ± 0.118 | 9.701 ± 0.703 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 36.516 ± 2.221 | 11.771 ± 0.454 | 27.436 ± 1.686 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.216 ± 0.213 | 4.104 ± 0.307 | 13.779 ± 0.826 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 35.356 ± 4.535 | 18.377 ± 0.246 | 45.701 ± 3.240 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 8.060 ± 0.367 | 4.023 ± 0.133 | 13.822 ± 0.594 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 34.773 ± 0.563 | 18.073 ± 0.117 | 59.183 ± 6.251 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.255 ± 0.085 | 1.533 ± 0.073 | 4.011 ± 0.281 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 9.284 ± 0.207 | 3.131 ± 0.253 | 10.353 ± 0.116 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.593 ± 0.058 | 1.644 ± 0.068 | 5.112 ± 0.290 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 6.582 ± 0.210 | 4.807 ± 0.109 | 10.946 ± 0.226 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 21.576 ± 0.260 | 20.167 ± 0.380 | 52.543 ± 0.999 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 99.696 ± 7.265 | 77.024 ± 1.241 | 164.458 ± 5.072 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 22.799 ± 0.730 | 16.296 ± 0.311 | 50.039 ± 0.850 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 104.809 ± 10.591 | 63.613 ± 1.327 | 163.232 ± 2.812 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.058 ± 0.009 | 0.314 ± 0.039 | 1.312 ± 0.056 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.066 ± 0.014 | 0.331 ± 0.068 | 1.391 ± 0.051 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.078 ± 0.008 | 0.335 ± 0.067 | 1.384 ± 0.077 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.064 ± 0.010 | 0.308 ± 0.009 | 3.044 ± 0.162 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.066 ± 0.006 | 0.328 ± 0.051 | 3.084 ± 0.184 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.090 ± 0.013 | 0.322 ± 0.009 | 3.944 ± 0.830 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.118 ± 0.006 | 0.526 ± 0.052 | 1.836 ± 0.095 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.116 ± 0.005 | 0.510 ± 0.022 | 1.890 ± 0.098 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.132 ± 0.014 | 0.504 ± 0.042 | 2.395 ± 0.027 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.111 ± 0.004 | 0.452 ± 0.021 | 4.280 ± 1.495 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.116 ± 0.006 | 0.447 ± 0.015 | 4.650 ± 1.019 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.118 ± 0.007 | 0.456 ± 0.018 | 5.007 ± 2.763 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.170 ± 0.009 | 0.411 ± 0.069 | 2.712 ± 0.167 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.184 ± 0.006 | 0.392 ± 0.032 | 2.395 ± 0.125 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.192 ± 0.016 | 0.418 ± 0.098 | 2.327 ± 0.025 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.116 ± 0.007 | 0.419 ± 0.019 | 4.549 ± 2.401 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.122 ± 0.011 | 0.420 ± 0.024 | 4.674 ± 1.879 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.126 ± 0.007 | 0.456 ± 0.026 | 4.656 ± 1.918 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.044 ± 0.005 | 0.658 ± 0.005 | 1.961 ± 0.119 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.046 ± 0.004 | 0.667 ± 0.088 | 1.979 ± 0.084 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.048 ± 0.005 | 0.661 ± 0.011 | 2.145 ± 0.098 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.074 ± 0.002 | 0.432 ± 0.015 | 4.130 ± 0.960 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.077 ± 0.008 | 0.438 ± 0.013 | 4.121 ± 0.945 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.079 ± 0.005 | 0.441 ± 0.011 | 4.174 ± 0.926 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.039 ± 0.002 | 0.422 ± 0.019 | 1.610 ± 0.073 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.053 ± 0.002 | 0.426 ± 0.015 | 1.763 ± 0.039 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.069 ± 0.003 | 0.447 ± 0.013 | 1.881 ± 0.045 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.077 ± 0.004 | 0.344 ± 0.025 | 4.037 ± 1.462 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.084 ± 0.005 | 0.337 ± 0.010 | 3.968 ± 1.391 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.100 ± 0.009 | 0.362 ± 0.012 | 4.107 ± 1.625 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

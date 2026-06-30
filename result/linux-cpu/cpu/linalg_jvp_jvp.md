# Linux CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260630_115240`, raw run `data/results/linux-cpu/cpu/einsum/20260630_115240`
- Threads 4: timestamp `20260630_115451`, raw run `data/results/linux-cpu/cpu/einsum/20260630_115451`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260630_115240`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_115240`.

- tenferro-rs commit: `b4a68eeec7795817e45282eb8cac5082b0f4e03d`

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

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cu130`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/terasaki/work/atelierarith/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_115240/cpu_ops_t1_20260630_115240.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_115240/linalg_jvp_vjp_t1_20260630_115240.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.416 ± 0.600 | 11.892 ± 0.162 | 35.632 ± 1.119 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 91.820 ± 2.289 | 71.521 ± 2.477 | 115.636 ± 1.073 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 20.136 ± 0.410 | 9.822 ± 0.408 | 34.772 ± 0.703 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 122.166 ± 2.166 | 52.650 ± 0.688 | 114.664 ± 3.369 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.297 ± 0.173 | 11.785 ± 0.343 | 11.053 ± 2.262 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 47.637 ± 0.811 | 68.319 ± 0.678 | 49.245 ± 1.056 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 14.715 ± 0.197 | 5.394 ± 0.187 | 16.301 ± 1.746 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 100.170 ± 1.044 | 38.268 ± 0.317 | 63.349 ± 4.900 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 10.836 ± 0.273 | 7.800 ± 0.272 | 19.037 ± 1.341 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 67.825 ± 5.568 | 51.729 ± 0.952 | 69.756 ± 11.466 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 16.973 ± 0.102 | 7.630 ± 0.276 | 19.704 ± 0.261 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 112.517 ± 2.066 | 50.719 ± 0.662 | 101.136 ± 3.937 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.257 ± 0.109 | 1.921 ± 0.166 | 5.124 ± 0.075 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.650 ± 0.180 | 5.710 ± 0.286 | 15.804 ± 0.124 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.194 ± 0.113 | 2.095 ± 0.012 | 5.814 ± 0.332 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.229 ± 0.159 | 9.125 ± 0.320 | 16.592 ± 0.252 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 27.069 ± 0.204 | 24.587 ± 0.490 | 61.129 ± 2.457 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 160.266 ± 1.606 | 143.883 ± 3.277 | 205.540 ± 3.129 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 29.571 ± 0.377 | 18.817 ± 0.396 | 60.865 ± 2.403 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 177.499 ± 0.694 | 114.718 ± 1.257 | 204.838 ± 5.364 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.038 ± 0.001 | 0.359 ± 0.023 | 1.411 ± 0.054 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.045 ± 0.001 | 0.362 ± 0.010 | 1.423 ± 0.057 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.058 ± 0.020 | 0.379 ± 0.027 | 1.376 ± 0.051 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.073 ± 0.001 | 0.349 ± 0.022 | 3.142 ± 0.316 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.079 ± 0.001 | 0.357 ± 0.013 | 3.284 ± 0.370 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.089 ± 0.002 | 0.362 ± 0.010 | 3.762 ± 0.604 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.072 ± 0.003 | 0.503 ± 0.027 | 1.866 ± 0.059 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.074 ± 0.002 | 0.502 ± 0.096 | 1.887 ± 0.070 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.081 ± 0.001 | 0.513 ± 0.016 | 3.052 ± 0.111 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.102 ± 0.002 | 0.450 ± 0.011 | 4.168 ± 1.065 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.103 ± 0.002 | 0.460 ± 0.036 | 4.204 ± 1.212 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.109 ± 0.002 | 0.461 ± 0.039 | 4.995 ± 2.698 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.110 ± 0.001 | 0.404 ± 0.016 | 3.119 ± 0.570 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.114 ± 0.020 | 0.428 ± 0.075 | 2.343 ± 0.497 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.127 ± 0.015 | 0.402 ± 0.009 | 2.109 ± 0.079 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.173 ± 0.005 | 0.433 ± 0.013 | 4.335 ± 1.436 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.183 ± 0.007 | 0.443 ± 0.051 | 4.228 ± 1.394 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.190 ± 0.013 | 0.442 ± 0.049 | 4.852 ± 2.723 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.028 ± 0.009 | 0.733 ± 0.080 | 2.026 ± 0.358 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.025 ± 0.001 | 0.736 ± 0.010 | 2.220 ± 0.092 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.028 ± 0.001 | 0.736 ± 0.017 | 2.114 ± 0.046 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.034 ± 0.003 | 0.495 ± 0.016 | 3.249 ± 1.044 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.036 ± 0.001 | 0.496 ± 0.012 | 3.715 ± 1.404 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.040 ± 0.001 | 0.499 ± 0.022 | 4.165 ± 1.044 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.033 ± 0.002 | 0.487 ± 0.053 | 1.604 ± 0.087 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.036 ± 0.001 | 0.481 ± 0.011 | 1.752 ± 0.045 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.047 ± 0.002 | 0.503 ± 0.015 | 1.662 ± 0.059 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.056 ± 0.002 | 0.383 ± 0.014 | 3.980 ± 1.054 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.061 ± 0.001 | 0.382 ± 0.015 | 3.985 ± 1.215 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.072 ± 0.001 | 0.394 ± 0.012 | 4.084 ± 1.698 |

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
- Timestamp: `20260630_115451`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260630_115451`.

- tenferro-rs commit: `b4a68eeec7795817e45282eb8cac5082b0f4e03d`

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

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cu130`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/terasaki/work/atelierarith/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260630_115451/cpu_ops_t4_20260630_115451.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260630_115451/linalg_jvp_vjp_t4_20260630_115451.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 17.562 ± 0.406 | 7.550 ± 0.816 | 29.676 ± 0.315 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 83.084 ± 0.599 | 28.204 ± 0.773 | 88.463 ± 1.190 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 23.225 ± 0.201 | 6.267 ± 0.231 | 28.625 ± 0.249 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 107.916 ± 10.750 | 22.956 ± 0.520 | 89.280 ± 1.159 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 11.452 ± 1.530 | 4.319 ± 0.222 | 8.191 ± 1.812 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 42.527 ± 7.572 | 27.118 ± 2.308 | 21.213 ± 3.953 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 15.960 ± 2.880 | 2.729 ± 0.267 | 11.530 ± 1.093 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 75.934 ± 16.051 | 12.069 ± 0.410 | 31.635 ± 3.507 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 10.836 ± 0.594 | 4.732 ± 0.216 | 13.759 ± 0.522 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 42.496 ± 0.633 | 19.288 ± 0.599 | 45.932 ± 2.978 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 15.530 ± 0.302 | 4.171 ± 0.227 | 14.805 ± 2.502 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 62.364 ± 6.857 | 18.982 ± 0.430 | 61.511 ± 3.953 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 2.583 ± 0.071 | 1.687 ± 0.019 | 4.166 ± 0.082 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 11.925 ± 0.369 | 3.327 ± 0.282 | 10.450 ± 0.215 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 2.217 ± 0.114 | 1.921 ± 0.657 | 5.440 ± 0.875 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 8.824 ± 0.207 | 4.692 ± 0.229 | 11.172 ± 0.140 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 35.773 ± 2.082 | 20.998 ± 0.986 | 52.681 ± 1.661 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 131.426 ± 0.713 | 78.935 ± 1.597 | 165.953 ± 2.821 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 34.263 ± 0.303 | 18.113 ± 1.479 | 51.090 ± 0.989 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 142.755 ± 1.859 | 66.075 ± 2.424 | 162.609 ± 2.364 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.110 ± 0.026 | 0.358 ± 0.021 | 1.282 ± 0.026 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.129 ± 0.012 | 0.412 ± 0.042 | 1.368 ± 0.061 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.143 ± 0.031 | 0.406 ± 0.047 | 1.497 ± 0.060 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.206 ± 0.009 | 0.390 ± 0.054 | 3.278 ± 0.602 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.230 ± 0.008 | 0.404 ± 0.058 | 3.022 ± 0.407 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.251 ± 0.009 | 0.375 ± 0.020 | 4.073 ± 1.185 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.220 ± 0.010 | 0.623 ± 0.036 | 1.918 ± 0.065 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.234 ± 0.010 | 0.578 ± 0.031 | 1.813 ± 0.049 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.256 ± 0.024 | 0.606 ± 0.064 | 2.978 ± 0.134 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.335 ± 0.014 | 0.565 ± 0.047 | 4.238 ± 1.022 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.352 ± 0.012 | 0.527 ± 0.033 | 4.255 ± 1.308 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.381 ± 0.019 | 0.548 ± 0.044 | 4.967 ± 2.823 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.282 ± 0.024 | 0.444 ± 0.039 | 3.051 ± 0.347 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.313 ± 0.021 | 0.468 ± 0.027 | 2.541 ± 0.124 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.332 ± 0.019 | 0.463 ± 0.051 | 2.102 ± 0.080 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.537 ± 0.021 | 0.495 ± 0.022 | 4.406 ± 1.902 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.581 ± 0.017 | 0.513 ± 0.025 | 4.256 ± 1.767 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.612 ± 0.048 | 0.524 ± 0.037 | 4.767 ± 2.567 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.129 ± 0.012 | 0.791 ± 0.067 | 2.051 ± 0.407 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.132 ± 0.012 | 0.815 ± 0.094 | 2.111 ± 0.120 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.137 ± 0.011 | 0.786 ± 0.114 | 2.446 ± 0.049 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.213 ± 0.014 | 0.509 ± 0.023 | 3.336 ± 1.082 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.220 ± 0.014 | 0.515 ± 0.030 | 4.073 ± 0.959 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.236 ± 0.013 | 0.498 ± 0.010 | 4.304 ± 1.019 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.085 ± 0.013 | 0.473 ± 0.044 | 1.685 ± 0.067 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.098 ± 0.008 | 0.525 ± 0.085 | 1.768 ± 0.052 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.121 ± 0.010 | 0.520 ± 0.068 | 1.676 ± 0.073 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.154 ± 0.016 | 0.396 ± 0.054 | 4.177 ± 1.536 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.172 ± 0.015 | 0.423 ± 0.037 | 4.153 ± 1.240 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.200 ± 0.030 | 0.421 ± 0.043 | 4.152 ± 1.640 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

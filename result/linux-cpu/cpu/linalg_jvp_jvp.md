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

- Threads 1: timestamp `20260716_093114`, raw run `data/results/linux-cpu/cpu/einsum/20260716_093114`
- Threads 4: timestamp `20260716_094143`, raw run `data/results/linux-cpu/cpu/einsum/20260716_094143`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260716_093114`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260716_093114`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_093114/cpu_ops_t1_20260716_093114.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_093114/linalg_jvp_vjp_t1_20260716_093114.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 14.303 ± 0.113 | 11.681 ± 0.015 | 35.339 ± 1.267 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 74.795 ± 1.037 | 70.731 ± 0.374 | 113.888 ± 1.639 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 13.786 ± 0.124 | 9.698 ± 0.122 | 34.923 ± 0.691 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 73.200 ± 0.300 | 52.902 ± 0.481 | 115.237 ± 2.175 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 8.243 ± 0.245 | 8.979 ± 0.031 | 11.449 ± 2.219 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 39.612 ± 4.973 | 66.439 ± 0.465 | 45.331 ± 4.251 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.090 ± 0.147 | 5.197 ± 0.014 | 16.703 ± 2.191 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 42.511 ± 4.115 | 37.232 ± 0.070 | 54.487 ± 3.936 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 11.345 ± 0.224 | 7.612 ± 0.014 | 18.451 ± 0.440 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 59.124 ± 4.666 | 50.292 ± 0.240 | 79.485 ± 2.871 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.644 ± 0.370 | 7.349 ± 0.023 | 19.576 ± 1.522 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 61.414 ± 0.913 | 49.240 ± 0.438 | 101.327 ± 3.223 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.532 ± 0.022 | 1.692 ± 0.090 | 5.176 ± 0.091 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.962 ± 0.041 | 5.575 ± 0.009 | 15.672 ± 0.184 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.210 ± 0.053 | 2.044 ± 0.004 | 6.019 ± 0.350 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.639 ± 0.053 | 8.953 ± 0.177 | 16.410 ± 0.217 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 23.130 ± 0.212 | 21.814 ± 0.094 | 61.816 ± 2.200 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 127.844 ± 1.074 | 139.568 ± 1.040 | 204.550 ± 5.044 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 22.687 ± 0.151 | 18.084 ± 0.127 | 59.691 ± 2.741 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 126.636 ± 0.914 | 112.698 ± 0.426 | 202.147 ± 3.964 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.083 ± 0.007 | 0.331 ± 0.011 | 1.414 ± 0.052 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.091 ± 0.016 | 0.338 ± 0.012 | 1.409 ± 0.062 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.101 ± 0.006 | 0.352 ± 0.009 | 1.494 ± 0.066 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.097 ± 0.021 | 0.314 ± 0.010 | 3.198 ± 0.356 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.110 ± 0.015 | 0.323 ± 0.011 | 3.101 ± 0.190 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.109 ± 0.013 | 0.345 ± 0.047 | 4.126 ± 1.171 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.087 ± 0.005 | 0.465 ± 0.012 | 1.944 ± 0.109 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.089 ± 0.005 | 0.467 ± 0.013 | 1.930 ± 0.086 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.091 ± 0.005 | 0.507 ± 0.049 | 3.163 ± 0.317 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.092 ± 0.006 | 0.406 ± 0.010 | 4.573 ± 1.194 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.095 ± 0.013 | 0.409 ± 0.012 | 4.140 ± 0.788 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.098 ± 0.007 | 0.429 ± 0.033 | 5.069 ± 2.694 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.228 ± 0.020 | 0.369 ± 0.013 | 3.144 ± 0.409 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.222 ± 0.018 | 0.370 ± 0.011 | 2.328 ± 0.094 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.227 ± 0.021 | 0.371 ± 0.008 | 2.205 ± 0.085 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.521 ± 0.052 | 0.393 ± 0.009 | 4.169 ± 1.270 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.499 ± 0.027 | 0.398 ± 0.008 | 5.066 ± 1.637 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.496 ± 0.005 | 0.399 ± 0.010 | 4.905 ± 2.651 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.062 ± 0.007 | 0.690 ± 0.011 | 2.178 ± 0.101 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.058 ± 0.006 | 0.691 ± 0.007 | 1.984 ± 0.094 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.059 ± 0.005 | 0.697 ± 0.026 | 2.342 ± 0.034 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.094 ± 0.009 | 0.445 ± 0.010 | 4.114 ± 1.107 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.095 ± 0.010 | 0.448 ± 0.010 | 3.458 ± 1.396 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.098 ± 0.008 | 0.454 ± 0.011 | 4.225 ± 1.188 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.052 ± 0.002 | 0.444 ± 0.014 | 1.757 ± 0.088 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.060 ± 0.005 | 0.444 ± 0.013 | 1.754 ± 0.037 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.069 ± 0.007 | 0.464 ± 0.010 | 1.671 ± 0.021 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.063 ± 0.004 | 0.349 ± 0.014 | 4.076 ± 1.308 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.068 ± 0.004 | 0.350 ± 0.009 | 4.064 ± 1.255 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.081 ± 0.006 | 0.365 ± 0.010 | 4.222 ± 1.742 |

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
- Timestamp: `20260716_094143`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260716_094143`.

- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260716_094143/cpu_ops_t4_20260716_094143.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260716_094143/linalg_jvp_vjp_t4_20260716_094143.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 12.999 ± 0.142 | 7.934 ± 1.273 | 29.582 ± 0.231 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 54.316 ± 0.421 | 27.150 ± 0.149 | 88.088 ± 2.082 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 12.808 ± 0.141 | 5.991 ± 0.065 | 28.529 ± 0.296 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 53.057 ± 0.327 | 22.209 ± 0.351 | 88.546 ± 1.743 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.982 ± 0.288 | 3.978 ± 0.042 | 8.406 ± 1.732 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 20.572 ± 5.304 | 23.588 ± 0.222 | 24.775 ± 1.083 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.707 ± 0.285 | 2.615 ± 0.034 | 10.288 ± 1.272 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 24.192 ± 5.713 | 11.464 ± 0.225 | 33.896 ± 2.968 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 8.679 ± 0.164 | 4.491 ± 0.025 | 13.891 ± 0.900 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 33.099 ± 4.864 | 18.496 ± 0.089 | 45.698 ± 1.965 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 8.647 ± 0.268 | 3.975 ± 0.304 | 14.243 ± 1.294 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 33.486 ± 0.552 | 18.291 ± 0.269 | 60.215 ± 1.436 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.631 ± 0.034 | 1.559 ± 0.039 | 4.118 ± 0.054 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 6.149 ± 0.140 | 3.133 ± 0.101 | 10.627 ± 0.069 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.208 ± 0.080 | 1.692 ± 0.036 | 5.564 ± 0.512 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 4.451 ± 0.099 | 4.519 ± 0.023 | 11.013 ± 0.468 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 20.698 ± 0.097 | 17.709 ± 0.103 | 52.397 ± 1.215 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 94.043 ± 1.448 | 76.471 ± 0.346 | 163.982 ± 3.683 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.365 ± 0.127 | 16.040 ± 0.536 | 50.794 ± 1.211 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 92.540 ± 0.438 | 63.637 ± 0.763 | 162.770 ± 2.696 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.115 ± 0.014 | 0.335 ± 0.012 | 1.441 ± 0.067 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.117 ± 0.011 | 0.342 ± 0.027 | 1.472 ± 0.049 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.126 ± 0.015 | 0.355 ± 0.039 | 1.586 ± 0.065 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.119 ± 0.016 | 0.317 ± 0.010 | 3.374 ± 0.408 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.129 ± 0.014 | 0.334 ± 0.011 | 3.377 ± 0.430 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.144 ± 0.014 | 0.336 ± 0.011 | 4.016 ± 0.956 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.113 ± 0.014 | 0.514 ± 0.039 | 1.999 ± 0.064 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.121 ± 0.011 | 0.515 ± 0.043 | 2.013 ± 0.115 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.123 ± 0.019 | 0.519 ± 0.020 | 3.196 ± 0.244 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.117 ± 0.013 | 0.457 ± 0.013 | 4.575 ± 1.075 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.117 ± 0.016 | 0.462 ± 0.025 | 4.597 ± 1.102 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.126 ± 0.017 | 0.459 ± 0.012 | 5.077 ± 2.684 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.281 ± 0.029 | 0.403 ± 0.019 | 2.789 ± 0.150 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.284 ± 0.021 | 0.402 ± 0.019 | 2.362 ± 0.129 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.301 ± 0.026 | 0.406 ± 0.020 | 2.186 ± 0.035 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.591 ± 0.038 | 0.433 ± 0.012 | 4.811 ± 2.549 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.624 ± 0.042 | 0.439 ± 0.015 | 4.742 ± 1.768 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.633 ± 0.101 | 0.441 ± 0.017 | 4.883 ± 2.616 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.073 ± 0.005 | 0.692 ± 0.008 | 2.148 ± 0.095 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.077 ± 0.012 | 0.695 ± 0.008 | 2.104 ± 0.036 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.073 ± 0.010 | 0.694 ± 0.013 | 2.004 ± 0.089 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.116 ± 0.010 | 0.446 ± 0.010 | 4.203 ± 1.162 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.116 ± 0.011 | 0.449 ± 0.009 | 4.392 ± 1.500 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.119 ± 0.011 | 0.453 ± 0.008 | 4.201 ± 0.927 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.067 ± 0.003 | 0.436 ± 0.015 | 1.716 ± 0.079 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.071 ± 0.005 | 0.436 ± 0.009 | 1.873 ± 0.064 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.084 ± 0.009 | 0.456 ± 0.011 | 1.653 ± 0.066 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.079 ± 0.006 | 0.348 ± 0.015 | 4.112 ± 1.538 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.081 ± 0.014 | 0.349 ± 0.009 | 4.061 ± 1.278 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.099 ± 0.010 | 0.366 ± 0.010 | 4.187 ± 1.806 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

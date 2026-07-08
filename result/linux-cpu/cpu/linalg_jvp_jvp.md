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

- Threads 1: timestamp `20260708_221713`, raw run `data/results/linux-cpu/cpu/einsum/20260708_221713`
- Threads 4: timestamp `20260708_222212`, raw run `data/results/linux-cpu/cpu/einsum/20260708_222212`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260708_221713`

Latest run: `./scripts/run_all.sh 1`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260708_221713`.

- tenferro-rs commit: `4ee8f9857b4b5f599c3aad3fbd992375f934f1f7`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260708_221713/cpu_ops_t1_20260708_221713.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260708_221713/linalg_jvp_vjp_t1_20260708_221713.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 20.089 ± 0.224 | 22.426 ± 1.299 | 26.198 ± 2.952 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 120.253 ± 3.483 | 134.868 ± 2.222 | 122.466 ± 8.326 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 19.299 ± 0.249 | 18.075 ± 0.379 | 25.732 ± 2.137 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 113.032 ± 2.779 | 96.767 ± 2.296 | 119.373 ± 6.330 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 13.242 ± 0.437 | 21.662 ± 0.383 | 13.354 ± 3.858 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 71.222 ± 7.515 | 164.384 ± 6.910 | 49.559 ± 5.168 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 12.258 ± 0.273 | 12.695 ± 0.262 | 16.021 ± 3.810 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 71.741 ± 7.360 | 84.292 ± 2.192 | 61.492 ± 6.996 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 15.726 ± 0.542 | 17.116 ± 0.836 | 17.949 ± 3.786 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 101.790 ± 5.245 | 116.623 ± 4.618 | 84.440 ± 12.325 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 15.535 ± 0.303 | 15.512 ± 0.575 | 20.177 ± 4.223 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 102.336 ± 2.009 | 105.984 ± 2.224 | 102.780 ± 9.267 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 2.591 ± 0.106 | 4.288 ± 0.467 | 5.248 ± 1.662 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 13.987 ± 0.547 | 13.926 ± 1.948 | 15.686 ± 2.955 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.646 ± 0.046 | 4.917 ± 0.322 | 5.975 ± 1.309 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 10.301 ± 0.402 | 20.258 ± 0.772 | 17.530 ± 5.252 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 33.243 ± 0.597 | 42.055 ± 1.029 | 49.648 ± 3.452 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 203.577 ± 5.207 | 268.203 ± 1.864 | 249.892 ± 14.790 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 32.413 ± 0.206 | 33.583 ± 1.733 | 46.582 ± 5.373 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 204.869 ± 4.514 | 213.206 ± 1.733 | 238.223 ± 10.107 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.080 ± 0.009 | 0.642 ± 0.027 | 1.494 ± 0.021 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.083 ± 0.006 | 0.645 ± 0.018 | 1.516 ± 0.020 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.110 ± 0.009 | 0.668 ± 0.028 | 1.904 ± 0.135 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.090 ± 0.010 | 0.584 ± 0.049 | 3.933 ± 0.149 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.092 ± 0.014 | 0.585 ± 0.026 | 3.720 ± 0.374 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.113 ± 0.007 | 0.607 ± 0.022 | 3.892 ± 0.465 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.092 ± 0.005 | 0.883 ± 0.040 | 2.170 ± 0.046 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.081 ± 0.009 | 0.906 ± 0.051 | 2.658 ± 0.182 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.084 ± 0.010 | 0.917 ± 0.024 | 2.357 ± 0.184 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.091 ± 0.008 | 0.749 ± 0.026 | 5.835 ± 0.470 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.078 ± 0.004 | 0.743 ± 0.040 | 6.396 ± 0.566 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.082 ± 0.006 | 0.776 ± 0.012 | 5.845 ± 0.660 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.206 ± 0.027 | 0.716 ± 0.014 | 2.343 ± 0.192 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.211 ± 0.026 | 0.694 ± 0.035 | 2.188 ± 0.105 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.210 ± 0.017 | 0.707 ± 0.046 | 2.304 ± 0.126 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.460 ± 0.021 | 0.723 ± 0.052 | 5.793 ± 0.557 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.482 ± 0.017 | 0.733 ± 0.043 | 6.000 ± 0.265 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.486 ± 0.047 | 0.727 ± 0.046 | 6.218 ± 0.281 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.052 ± 0.005 | 1.283 ± 0.045 | 1.867 ± 0.090 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.047 ± 0.006 | 1.314 ± 0.078 | 1.765 ± 0.239 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.053 ± 0.007 | 1.406 ± 0.149 | 1.855 ± 0.134 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.074 ± 0.005 | 0.822 ± 0.034 | 4.348 ± 0.558 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.067 ± 0.008 | 0.831 ± 0.025 | 3.937 ± 0.345 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.074 ± 0.005 | 0.800 ± 0.021 | 4.681 ± 0.738 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.066 ± 0.006 | 0.843 ± 0.031 | 1.218 ± 0.070 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.077 ± 0.004 | 0.857 ± 0.025 | 1.630 ± 0.108 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.104 ± 0.018 | 0.867 ± 0.033 | 1.793 ± 0.110 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.075 ± 0.004 | 0.609 ± 0.030 | 3.737 ± 0.207 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.084 ± 0.007 | 0.629 ± 0.045 | 4.311 ± 0.539 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.128 ± 0.008 | 0.660 ± 0.027 | 4.297 ± 0.251 |

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
- Timestamp: `20260708_222212`

Latest run: `./scripts/run_all.sh 4`.

Reproduce: `./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260708_222212`.

- tenferro-rs commit: `4ee8f9857b4b5f599c3aad3fbd992375f934f1f7`

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

- CSV: `data/results/linux-cpu/cpu/einsum/20260708_222212/cpu_ops_t4_20260708_222212.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260708_222212/linalg_jvp_vjp_t4_20260708_222212.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.910 ± 0.235 | 8.110 ± 2.694 | 30.336 ± 0.443 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 33.312 ± 0.378 | 31.811 ± 0.308 | 88.680 ± 1.132 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.866 ± 0.061 | 6.959 ± 0.128 | 29.402 ± 1.188 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 31.594 ± 0.454 | 23.538 ± 0.163 | 89.244 ± 2.107 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.754 ± 0.366 | 4.591 ± 0.143 | 8.311 ± 1.658 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 21.005 ± 6.500 | 28.956 ± 0.515 | 21.449 ± 0.599 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 6.777 ± 0.301 | 2.895 ± 0.029 | 11.130 ± 1.289 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 22.663 ± 5.017 | 14.004 ± 0.378 | 25.604 ± 1.960 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.408 ± 0.787 | 4.745 ± 0.101 | 13.772 ± 1.246 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 30.844 ± 3.880 | 21.889 ± 0.316 | 45.593 ± 0.436 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 8.005 ± 0.509 | 4.538 ± 0.058 | 15.804 ± 1.621 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 29.729 ± 0.892 | 21.357 ± 0.272 | 54.543 ± 9.660 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.829 ± 0.123 | 1.621 ± 0.022 | 4.117 ± 0.090 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 6.285 ± 0.259 | 3.657 ± 0.229 | 10.875 ± 0.421 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.035 ± 0.092 | 1.748 ± 0.010 | 5.154 ± 0.399 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.805 ± 0.201 | 5.234 ± 0.234 | 11.212 ± 0.209 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 19.573 ± 1.500 | 20.204 ± 0.145 | 53.026 ± 1.367 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 79.627 ± 1.067 | 89.747 ± 0.436 | 167.931 ± 7.501 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 17.704 ± 0.116 | 18.161 ± 0.153 | 52.124 ± 2.014 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 77.549 ± 0.591 | 74.305 ± 0.446 | 161.924 ± 4.294 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.094 ± 0.005 | 0.391 ± 0.010 | 1.069 ± 0.020 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.072 ± 0.003 | 0.402 ± 0.013 | 1.169 ± 0.031 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.123 ± 0.010 | 0.414 ± 0.013 | 1.128 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.104 ± 0.023 | 0.381 ± 0.014 | 2.972 ± 0.091 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.077 ± 0.008 | 0.392 ± 0.012 | 3.307 ± 0.435 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.134 ± 0.012 | 0.407 ± 0.014 | 3.842 ± 1.111 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.079 ± 0.005 | 0.574 ± 0.016 | 1.549 ± 0.015 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.080 ± 0.005 | 0.577 ± 0.011 | 1.736 ± 0.034 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.133 ± 0.008 | 0.581 ± 0.012 | 2.709 ± 0.046 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.080 ± 0.002 | 0.516 ± 0.009 | 4.445 ± 1.284 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.077 ± 0.003 | 0.517 ± 0.009 | 4.300 ± 1.384 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.135 ± 0.005 | 0.523 ± 0.009 | 4.559 ± 2.618 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.170 ± 0.008 | 0.444 ± 0.013 | 2.755 ± 0.252 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.225 ± 0.016 | 0.443 ± 0.010 | 2.237 ± 0.024 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.262 ± 0.013 | 0.449 ± 0.011 | 2.226 ± 0.034 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.629 ± 0.020 | 0.485 ± 0.013 | 4.546 ± 2.556 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.382 ± 0.022 | 0.495 ± 0.012 | 4.176 ± 2.231 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.576 ± 0.024 | 0.499 ± 0.011 | 4.586 ± 2.519 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.047 ± 0.002 | 0.830 ± 0.009 | 1.872 ± 0.015 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.047 ± 0.003 | 0.830 ± 0.007 | 1.860 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.055 ± 0.010 | 0.833 ± 0.004 | 2.008 ± 0.020 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.066 ± 0.008 | 0.540 ± 0.013 | 3.929 ± 1.248 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.068 ± 0.006 | 0.542 ± 0.011 | 3.953 ± 1.227 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.068 ± 0.010 | 0.545 ± 0.008 | 4.100 ± 1.120 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.048 ± 0.002 | 0.525 ± 0.018 | 1.511 ± 0.021 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.058 ± 0.002 | 0.530 ± 0.011 | 1.625 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.117 ± 0.002 | 0.555 ± 0.008 | 1.547 ± 0.024 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.053 ± 0.002 | 0.419 ± 0.016 | 3.745 ± 1.408 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.062 ± 0.004 | 0.417 ± 0.009 | 3.653 ± 1.260 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.123 ± 0.005 | 0.441 ± 0.011 | 3.799 ± 1.568 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

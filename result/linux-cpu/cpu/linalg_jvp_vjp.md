# Linux CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260728_013419 4:20260728_015909 permutation:20260728_021651`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260728_013419`, raw run `data/results/linux-cpu/cpu/einsum/20260728_013419`
- Threads 4: timestamp `20260728_015909`, raw run `data/results/linux-cpu/cpu/einsum/20260728_015909`

## Threads: 1

### CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `linux-cpu`
- Timestamp: `20260728_013419`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260728_013419`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260728_013419/cpu_ops_t1_20260728_013419.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260728_013419/linalg_jvp_vjp_t1_20260728_013419.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 9.064 ± 0.040 | 10.574 ± 0.027 | 17.013 ± 0.539 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 55.894 ± 2.239 | 74.686 ± 0.140 | 64.454 ± 1.766 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.763 ± 0.074 | 8.698 ± 0.010 | 16.673 ± 1.240 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 63.834 ± 0.594 | 52.969 ± 0.092 | 66.897 ± 5.224 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 6.668 ± 0.178 | 12.679 ± 0.011 | 6.192 ± 0.547 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 44.815 ± 0.384 | 86.364 ± 0.191 | 33.107 ± 4.292 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 6.377 ± 0.481 | 5.582 ± 0.008 | 6.278 ± 0.470 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 43.135 ± 0.334 | 47.325 ± 0.290 | 36.848 ± 8.133 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 8.529 ± 1.176 | 7.710 ± 0.023 | 7.877 ± 3.073 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 54.923 ± 2.992 | 61.072 ± 0.196 | 46.338 ± 1.181 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.916 ± 0.623 | 7.535 ± 0.014 | 11.114 ± 2.604 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 48.624 ± 0.539 | 60.115 ± 0.158 | 70.849 ± 12.535 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.422 ± 0.007 | 1.464 ± 0.006 | 2.130 ± 0.205 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 6.125 ± 0.039 | 6.015 ± 0.007 | 9.472 ± 0.841 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.378 ± 0.012 | 1.937 ± 0.019 | 2.644 ± 0.088 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 6.061 ± 0.024 | 9.949 ± 0.036 | 10.133 ± 0.290 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 14.250 ± 0.366 | 19.466 ± 0.081 | 30.285 ± 12.269 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 91.832 ± 1.144 | 141.249 ± 0.712 | 120.480 ± 3.988 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 13.945 ± 0.237 | 16.504 ± 0.018 | 31.983 ± 13.253 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 90.469 ± 3.817 | 115.386 ± 0.356 | 121.976 ± 16.647 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.181 ± 0.001 | 0.227 ± 0.011 | 0.579 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.184 ± 0.003 | 0.225 ± 0.005 | 0.584 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.188 ± 0.003 | 0.234 ± 0.005 | 0.589 ± 0.021 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.183 ± 0.004 | 0.225 ± 0.008 | 1.318 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.187 ± 0.004 | 0.228 ± 0.012 | 1.326 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.188 ± 0.004 | 0.233 ± 0.007 | 2.279 ± 0.752 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.180 ± 0.006 | 0.308 ± 0.011 | 0.829 ± 0.025 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.179 ± 0.002 | 0.305 ± 0.008 | 0.834 ± 0.016 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.201 ± 0.004 | 0.308 ± 0.008 | 0.816 ± 0.030 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.192 ± 0.002 | 0.296 ± 0.016 | 2.103 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.216 ± 0.003 | 0.287 ± 0.012 | 2.090 ± 0.019 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.192 ± 0.027 | 0.289 ± 0.013 | 2.085 ± 0.009 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.323 ± 0.007 | 0.246 ± 0.014 | 1.545 ± 0.252 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.323 ± 0.008 | 0.250 ± 0.013 | 1.700 ± 0.040 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.333 ± 0.004 | 0.250 ± 0.007 | 1.743 ± 0.043 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.329 ± 0.002 | 0.273 ± 0.023 | 1.950 ± 0.029 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.339 ± 0.042 | 0.273 ± 0.019 | 4.057 ± 0.531 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.336 ± 0.010 | 0.280 ± 0.010 | 4.161 ± 0.274 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.198 ± 0.003 | 0.496 ± 0.017 | 0.689 ± 0.025 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.225 ± 0.009 | 0.486 ± 0.013 | 0.695 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.227 ± 0.009 | 0.498 ± 0.009 | 0.705 ± 0.036 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.219 ± 0.004 | 0.316 ± 0.011 | 2.653 ± 0.027 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.220 ± 0.028 | 0.318 ± 0.015 | 2.668 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.225 ± 0.001 | 0.319 ± 0.015 | 1.440 ± 0.025 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.118 ± 0.002 | 0.301 ± 0.009 | 1.102 ± 0.043 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.124 ± 0.003 | 0.301 ± 0.007 | 1.113 ± 0.029 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.135 ± 0.011 | 0.310 ± 0.010 | 1.075 ± 0.038 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.106 ± 0.029 | 0.261 ± 0.020 | 2.944 ± 0.023 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.125 ± 0.004 | 0.248 ± 0.010 | 2.988 ± 0.039 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.120 ± 0.002 | 0.265 ± 0.013 | 3.057 ± 0.023 |

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
- Timestamp: `20260728_015909`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/linux-cpu/cpu/einsum/20260728_015909`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260728_015909/cpu_ops_t4_20260728_015909.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260728_015909/linalg_jvp_vjp_t4_20260728_015909.md`

#### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 62.996 ± 14.374 | 6.796 ± 0.222 | 14.493 ± 3.775 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 172.693 ± 8.039 | 33.858 ± 0.860 | 51.672 ± 11.005 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 70.433 ± 14.696 | 5.895 ± 0.079 | 14.597 ± 4.601 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 97.448 ± 1.793 | 22.678 ± 2.169 | 52.944 ± 12.229 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 42.591 ± 2.101 | 7.554 ± 1.166 | 4.014 ± 0.520 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 159.262 ± 8.610 | 32.516 ± 1.149 | 14.129 ± 1.227 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 43.311 ± 4.269 | 3.249 ± 0.031 | 7.244 ± 0.793 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 178.901 ± 11.545 | 18.069 ± 0.922 | 15.600 ± 1.149 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 56.400 ± 6.916 | 4.448 ± 0.420 | 7.295 ± 2.581 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 236.788 ± 4121.962 | 23.965 ± 1.402 | 36.223 ± 9.099 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 53.952 ± 1.687 | 5.001 ± 0.653 | 10.108 ± 1.445 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 228.478 ± 36.682 | 25.298 ± 0.863 | 56.087 ± 10.404 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 16.333 ± 8.153 | 1.240 ± 0.013 | 2.643 ± 0.266 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 100.197 ± 1.012 | 2.995 ± 0.029 | 6.948 ± 0.305 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 14.536 ± 12.643 | 1.545 ± 0.108 | 2.233 ± 0.363 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 80.277 ± 18.023 | 4.581 ± 0.035 | 6.799 ± 0.244 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 100.653 ± 22.089 | 14.089 ± 0.274 | 28.923 ± 11.240 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 190.369 ± 76.573 | 81.753 ± 7.483 | 104.181 ± 16.088 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 91.903 ± 6.826 | 13.093 ± 0.282 | 29.012 ± 11.053 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 186.015 ± 3.519 | 74.330 ± 2.723 | 100.685 ± 10.115 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.234 ± 0.040 | 0.193 ± 0.051 | 0.573 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.244 ± 0.007 | 0.193 ± 0.005 | 0.587 ± 0.021 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.243 ± 0.003 | 0.196 ± 0.008 | 1.061 ± 0.036 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.238 ± 0.004 | 0.192 ± 0.006 | 1.568 ± 1.074 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.238 ± 0.006 | 0.194 ± 0.009 | 1.314 ± 0.042 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.245 ± 0.003 | 0.236 ± 0.041 | 2.761 ± 0.031 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.225 ± 0.006 | 0.276 ± 0.015 | 0.818 ± 0.019 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.224 ± 0.015 | 0.279 ± 0.009 | 0.825 ± 0.024 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.237 ± 0.006 | 0.331 ± 0.070 | 0.830 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.238 ± 0.012 | 0.306 ± 0.049 | 2.054 ± 0.023 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.245 ± 0.007 | 0.261 ± 0.009 | 2.079 ± 0.017 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.245 ± 0.006 | 0.261 ± 0.010 | 2.111 ± 0.046 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.420 ± 0.012 | 0.262 ± 0.015 | 1.720 ± 0.040 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.427 ± 0.012 | 0.219 ± 0.014 | 1.537 ± 0.034 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.435 ± 0.006 | 0.266 ± 0.008 | 1.625 ± 0.407 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.421 ± 0.009 | 0.241 ± 0.015 | 3.931 ± 0.071 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.422 ± 0.016 | 0.244 ± 0.086 | 4.026 ± 0.444 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.434 ± 0.005 | 0.299 ± 0.039 | 4.173 ± 0.423 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.261 ± 0.031 | 0.420 ± 0.013 | 0.681 ± 0.024 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.255 ± 0.006 | 0.484 ± 0.068 | 0.700 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.256 ± 0.003 | 0.419 ± 0.113 | 0.694 ± 0.017 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.279 ± 0.008 | 0.312 ± 0.056 | 1.425 ± 0.030 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.277 ± 0.011 | 0.290 ± 0.091 | 1.430 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.286 ± 0.024 | 0.279 ± 0.030 | 1.426 ± 0.024 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.134 ± 0.023 | 0.251 ± 0.008 | 1.220 ± 0.051 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.143 ± 0.021 | 0.254 ± 0.008 | 1.121 ± 0.048 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.153 ± 0.020 | 0.263 ± 0.010 | 1.216 ± 0.066 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.138 ± 0.019 | 0.221 ± 0.008 | 2.878 ± 0.031 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.145 ± 0.005 | 0.211 ± 0.010 | 2.978 ± 0.032 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.157 ± 0.009 | 0.223 ± 0.017 | 2.996 ± 0.020 |

#### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

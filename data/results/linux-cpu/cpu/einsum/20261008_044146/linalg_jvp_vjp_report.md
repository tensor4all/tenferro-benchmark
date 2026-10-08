# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261008_044146`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261008_044146`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

## Thread Environment

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_t1_20261008_044146.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261008_044146/linalg_jvp_vjp_t1_20261008_044146.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 348.894 ± 0.537 | 493.351 ± 1.190 | 323.311 ± 0.574 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.925 ± 0.019 | 11.077 ± 0.037 | 8.384 ± 0.027 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 55.310 ± 0.214 | 74.532 ± 0.165 | 51.286 ± 0.247 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 350.559 ± 0.476 | 356.213 ± 0.625 | 320.925 ± 0.413 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.878 ± 0.023 | 8.978 ± 0.048 | 8.350 ± 0.064 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 55.276 ± 0.129 | 56.729 ± 0.082 | 51.124 ± 0.305 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 282.367 ± 0.360 | 609.271 ± 2.501 | 234.359 ± 2.491 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.342 ± 0.017 | 10.825 ± 0.076 | 4.599 ± 0.016 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 38.496 ± 0.209 | 82.831 ± 0.187 | 32.653 ± 0.091 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 276.074 ± 0.653 | 326.508 ± 5.939 | 235.357 ± 1.248 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.465 ± 0.006 | 5.832 ± 0.023 | 4.504 ± 0.022 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 38.204 ± 0.065 | 43.382 ± 0.117 | 31.662 ± 0.072 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 365.127 ± 0.494 | 418.926 ± 1.604 | 301.561 ± 1.265 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 7.699 ± 0.038 | 8.272 ± 0.018 | 6.457 ± 0.017 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 57.267 ± 0.091 | 66.265 ± 0.747 | 42.961 ± 0.350 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 360.289 ± 0.408 | 407.364 ± 2.039 | 331.027 ± 0.619 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.658 ± 0.014 | 8.287 ± 0.025 | 6.462 ± 0.041 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 54.552 ± 0.101 | 68.759 ± 0.876 | 47.262 ± 0.587 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 27.206 ± 0.092 | 36.963 ± 0.129 | 22.241 ± 0.089 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.720 ± 0.006 | 1.068 ± 0.017 | 0.610 ± 0.008 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.274 ± 0.066 | 5.398 ± 0.025 | 4.051 ± 0.020 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 26.804 ± 0.083 | 65.111 ± 0.487 | 22.017 ± 0.050 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.705 ± 0.006 | 1.564 ± 0.010 | 0.623 ± 0.003 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.216 ± 0.026 | 9.538 ± 0.026 | 4.087 ± 0.015 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 676.216 ± 3.145 | 908.650 ± 4.960 | 607.557 ± 3.247 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 16.134 ± 0.091 | 20.226 ± 0.068 | 14.537 ± 0.023 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 107.400 ± 0.993 | 148.129 ± 0.664 | 99.509 ± 0.661 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 661.978 ± 15.387 | 684.864 ± 6.642 | 607.308 ± 1.994 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 16.094 ± 0.039 | 16.906 ± 0.064 | 14.569 ± 0.050 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 106.149 ± 2.119 | 125.661 ± 0.489 | 99.615 ± 0.461 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.066 ± 0.000 | 0.124 ± 0.001 | 0.032 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.046 ± 0.000 | 0.097 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.126 ± 0.001 | 0.197 ± 0.003 | 0.102 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.047 ± 0.000 | 0.098 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.052 ± 0.000 | 0.105 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.068 ± 0.000 | 0.107 ± 0.001 | 0.033 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.046 ± 0.000 | 0.083 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.125 ± 0.000 | 0.171 ± 0.004 | 0.103 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.048 ± 0.000 | 0.086 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.053 ± 0.000 | 0.090 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.055 ± 0.000 | 0.175 ± 0.004 | 0.015 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.049 ± 0.000 | 0.158 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.079 ± 0.000 | 0.216 ± 0.004 | 0.030 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.049 ± 0.000 | 0.157 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.051 ± 0.000 | 0.161 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.056 ± 0.000 | 0.138 ± 0.001 | 0.014 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.049 ± 0.000 | 0.127 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.083 ± 0.000 | 0.166 ± 0.004 | 0.031 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.050 ± 0.000 | 0.127 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.051 ± 0.000 | 0.130 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.091 ± 0.001 | 0.124 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.081 ± 0.000 | 0.112 ± 0.003 | 0.011 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.119 ± 0.001 | 0.156 ± 0.002 | 0.039 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.082 ± 0.000 | 0.110 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.084 ± 0.001 | 0.114 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.094 ± 0.000 | 0.131 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.083 ± 0.000 | 0.118 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.125 ± 0.000 | 0.161 ± 0.004 | 0.039 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.084 ± 0.000 | 0.117 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.086 ± 0.000 | 0.119 ± 0.001 | 0.010 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.266 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.028 ± 0.000 | 0.259 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.038 ± 0.000 | 0.277 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.029 ± 0.000 | 0.257 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.029 ± 0.000 | 0.259 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.117 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.028 ± 0.000 | 0.112 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.037 ± 0.000 | 0.132 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.028 ± 0.000 | 0.111 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.029 ± 0.000 | 0.114 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.070 ± 0.000 | 0.210 ± 0.005 | 0.053 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.027 ± 0.000 | 0.147 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.174 ± 0.000 | 0.346 ± 0.009 | 0.158 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.030 ± 0.000 | 0.151 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.041 ± 0.000 | 0.167 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.069 ± 0.000 | 0.144 ± 0.001 | 0.054 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.026 ± 0.000 | 0.092 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.173 ± 0.001 | 0.269 ± 0.005 | 0.160 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.030 ± 0.000 | 0.096 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.041 ± 0.000 | 0.107 ± 0.001 | 0.018 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Collection recovery

Completed einsum measurements retained from the initial collection; CPU ops were recollected after the input-restoration fix. Component commits and original metadata are recorded in `run.yaml` and `einsum_run.yaml`. Exact commands: `data/results/amd-cpu/cpu/refresh/20261008_044136/commands_remaining.sh`.

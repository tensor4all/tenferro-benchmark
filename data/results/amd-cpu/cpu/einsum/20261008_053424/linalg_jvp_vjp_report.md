# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261008_053424`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261008_053424`.

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

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

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20261008_053424/cpu_ops_t4_20261008_053424.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261008_053424/linalg_jvp_vjp_t4_20261008_053424.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 110.595 ± 1.190 | 158.889 ± 0.821 | 142.499 ± 2.814 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 4.346 ± 0.042 | 4.765 ± 0.018 | 7.224 ± 0.444 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 19.898 ± 0.698 | 25.879 ± 0.099 | 30.225 ± 0.888 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 110.256 ± 0.489 | 112.010 ± 1.240 | 141.547 ± 1.556 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 4.343 ± 0.053 | 4.113 ± 0.017 | 6.640 ± 0.531 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 20.180 ± 0.389 | 19.726 ± 0.056 | 29.945 ± 1.244 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 81.038 ± 0.600 | 180.950 ± 0.557 | 70.229 ± 3.164 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.911 ± 0.025 | 4.312 ± 0.052 | 1.852 ± 0.325 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 12.071 ± 1.972 | 30.860 ± 0.381 | 10.272 ± 1.694 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 79.189 ± 0.214 | 92.926 ± 0.252 | 68.551 ± 1.687 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.167 ± 0.024 | 2.125 ± 0.016 | 1.753 ± 0.087 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 11.894 ± 0.095 | 14.338 ± 0.081 | 10.084 ± 0.540 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 113.904 ± 0.650 | 131.511 ± 0.602 | 112.847 ± 0.720 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 3.322 ± 0.030 | 3.344 ± 0.594 | 3.794 ± 0.308 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 22.380 ± 0.604 | 25.576 ± 0.157 | 19.457 ± 1.904 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 111.495 ± 0.286 | 126.004 ± 0.331 | 114.857 ± 2.341 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 3.191 ± 0.047 | 3.534 ± 0.046 | 3.846 ± 0.345 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 20.052 ± 0.347 | 25.700 ± 0.527 | 20.776 ± 2.378 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 8.244 ± 1.917 | 11.528 ± 0.113 | 9.718 ± 0.888 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.350 ± 0.002 | 0.624 ± 0.019 | 0.446 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.491 ± 0.079 | 3.028 ± 0.098 | 2.864 ± 0.761 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 7.920 ± 0.100 | 18.763 ± 0.164 | 9.293 ± 0.231 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.347 ± 0.009 | 0.701 ± 0.018 | 0.450 ± 0.020 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.437 ± 0.006 | 3.944 ± 0.017 | 2.570 ± 0.497 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 265.149 ± 2.585 | 344.704 ± 1.889 | 354.826 ± 6.893 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 10.761 ± 0.090 | 12.931 ± 0.063 | 12.295 ± 0.613 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 50.354 ± 1.528 | 68.240 ± 0.589 | 75.044 ± 1.311 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 266.303 ± 5.420 | 280.895 ± 4.921 | 360.760 ± 4.876 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 10.735 ± 0.024 | 12.043 ± 0.043 | 12.622 ± 1.063 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 50.639 ± 0.336 | 68.901 ± 1.123 | 75.148 ± 0.982 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.066 ± 0.000 | 0.123 ± 0.002 | 0.034 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.046 ± 0.000 | 0.095 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.130 ± 0.000 | 0.202 ± 0.003 | 0.103 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.048 ± 0.000 | 0.097 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.052 ± 0.000 | 0.104 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.067 ± 0.000 | 0.106 ± 0.002 | 0.034 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.047 ± 0.000 | 0.083 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.131 ± 0.000 | 0.176 ± 0.003 | 0.105 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.048 ± 0.000 | 0.084 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.052 ± 0.000 | 0.088 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.055 ± 0.000 | 0.179 ± 0.005 | 0.017 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.049 ± 0.000 | 0.163 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.088 ± 0.001 | 0.220 ± 0.004 | 0.030 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.049 ± 0.000 | 0.162 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.050 ± 0.000 | 0.163 ± 0.003 | 0.018 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.055 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.049 ± 0.000 | 0.127 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.094 ± 0.000 | 0.169 ± 0.005 | 0.030 ± 0.005 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.049 ± 0.000 | 0.129 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.050 ± 0.000 | 0.130 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.090 ± 0.000 | 0.128 ± 0.002 | 0.015 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.081 ± 0.000 | 0.113 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.121 ± 0.001 | 0.160 ± 0.005 | 0.041 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.082 ± 0.000 | 0.112 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.084 ± 0.000 | 0.115 ± 0.002 | 0.016 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.093 ± 0.000 | 0.132 ± 0.001 | 0.015 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.082 ± 0.000 | 0.118 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.123 ± 0.000 | 0.162 ± 0.004 | 0.041 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.084 ± 0.000 | 0.118 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.085 ± 0.000 | 0.119 ± 0.002 | 0.014 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.263 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.029 ± 0.000 | 0.255 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.038 ± 0.000 | 0.278 ± 0.003 | 0.017 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.028 ± 0.000 | 0.256 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.029 ± 0.000 | 0.256 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.115 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.028 ± 0.000 | 0.111 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.037 ± 0.000 | 0.132 ± 0.001 | 0.017 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.028 ± 0.000 | 0.110 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.028 ± 0.000 | 0.112 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.069 ± 0.000 | 0.206 ± 0.004 | 0.057 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.027 ± 0.000 | 0.144 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.183 ± 0.001 | 0.354 ± 0.003 | 0.163 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.030 ± 0.000 | 0.149 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.041 ± 0.000 | 0.165 ± 0.003 | 0.018 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.069 ± 0.000 | 0.141 ± 0.001 | 0.057 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.026 ± 0.000 | 0.090 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.182 ± 0.000 | 0.272 ± 0.003 | 0.176 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.030 ± 0.000 | 0.094 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.041 ± 0.000 | 0.104 ± 0.001 | 0.018 ± 0.002 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

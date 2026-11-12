# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

## Thread Environment

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_035559/cpu_ops_t8_20261113_035559.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | 200.217 ± 0.173 | 172.289 ± 0.332 | 133.519 ± 1.105 | 208.850 ± 10.311 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | 20.427 ± 0.115 | 15.361 ± 0.197 | 4.819 ± 0.037 | 11.752 ± 1.473 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | 46.356 ± 0.162 | 41.044 ± 0.039 | 21.747 ± 0.134 | 47.047 ± 5.961 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | 111.592 ± 0.339 | 172.219 ± 0.353 | 96.246 ± 1.156 | 202.972 ± 8.437 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | 10.387 ± 0.072 | 15.195 ± 0.046 | 4.305 ± 0.046 | 10.660 ± 2.244 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | 24.962 ± 0.036 | 40.871 ± 0.123 | 17.196 ± 0.045 | 47.552 ± 4.173 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | 132.679 ± 0.494 | 65.311 ± 0.170 | 144.877 ± 0.647 | 125.472 ± 15.277 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | 5.170 ± 0.053 | 2.384 ± 0.018 | 4.080 ± 0.052 | 5.679 ± 1.842 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | 20.136 ± 0.063 | 11.101 ± 0.043 | 27.513 ± 0.302 | 24.154 ± 5.345 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | 117.497 ± 0.721 | 67.514 ± 0.306 | 72.069 ± 0.326 | 134.454 ± 19.718 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | 3.521 ± 0.044 | 2.480 ± 0.006 | 2.088 ± 0.015 | 5.188 ± 1.414 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | 18.001 ± 0.105 | 11.795 ± 0.065 | 13.392 ± 0.261 | 23.335 ± 3.443 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | 166.417 ± 0.551 | 96.686 ± 0.319 | 112.283 ± 0.481 | 196.751 ± 17.818 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | 7.304 ± 0.136 | 3.492 ± 0.019 | 3.407 ± 0.025 | 9.983 ± 2.996 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | 26.913 ± 0.146 | 15.900 ± 0.076 | 22.242 ± 0.235 | 41.499 ± 8.487 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | 137.590 ± 0.515 | 95.811 ± 0.308 | 106.932 ± 0.333 | 196.039 ± 20.053 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | 3.959 ± 0.034 | 3.485 ± 0.037 | 3.667 ± 0.059 | 10.002 ± 2.643 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | 20.256 ± 0.083 | 15.696 ± 0.067 | 24.083 ± 0.098 | 43.270 ± 10.476 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | 16.085 ± 0.090 | 13.904 ± 0.157 | 9.124 ± 0.158 | 14.977 ± 1.396 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | 1.153 ± 0.020 | 0.748 ± 0.024 | 0.792 ± 0.029 | 0.590 ± 0.014 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | 4.055 ± 0.102 | 2.972 ± 0.051 | 2.566 ± 0.066 | 3.981 ± 0.209 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | 8.790 ± 0.044 | 13.657 ± 0.047 | 14.776 ± 0.086 | 14.341 ± 1.755 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | 0.633 ± 0.010 | 0.763 ± 0.021 | 0.891 ± 0.031 | 0.593 ± 0.014 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | 2.228 ± 0.122 | 2.875 ± 0.016 | 3.530 ± 0.033 | 3.630 ± 0.064 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | 424.268 ± 10.693 | 303.631 ± 0.826 | 324.881 ± 1.221 | 469.782 ± 40.800 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | 38.089 ± 0.107 | 27.550 ± 0.091 | 15.216 ± 0.100 | 19.751 ± 1.986 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | 81.423 ± 0.194 | 67.739 ± 0.105 | 72.928 ± 0.293 | 104.819 ± 5.395 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | 215.030 ± 0.640 | 327.511 ± 1.021 | 273.352 ± 2.462 | 472.943 ± 11.997 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | 19.105 ± 0.119 | 27.513 ± 0.033 | 14.221 ± 0.051 | 20.098 ± 0.933 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | 42.196 ± 0.098 | 67.777 ± 0.067 | 72.558 ± 0.194 | 104.024 ± 5.156 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | 0.313 ± 0.007 | 0.088 ± 0.000 | 0.156 ± 0.003 | 0.042 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | 0.236 ± 0.004 | 0.034 ± 0.000 | 0.122 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | 0.510 ± 0.010 | 0.240 ± 0.001 | 0.261 ± 0.005 | 0.132 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | 0.243 ± 0.003 | 0.038 ± 0.000 | 0.124 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | 0.261 ± 0.002 | 0.049 ± 0.000 | 0.132 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | 0.097 ± 0.001 | 0.086 ± 0.000 | 0.132 ± 0.002 | 0.043 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | 0.064 ± 0.001 | 0.033 ± 0.000 | 0.102 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | 0.194 ± 0.002 | 0.239 ± 0.000 | 0.226 ± 0.005 | 0.133 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | 0.067 ± 0.002 | 0.037 ± 0.000 | 0.105 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | 0.073 ± 0.001 | 0.048 ± 0.000 | 0.110 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | 0.946 ± 0.006 | 0.053 ± 0.000 | 0.230 ± 0.004 | 0.021 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | 0.896 ± 0.010 | 0.044 ± 0.000 | 0.211 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | 0.993 ± 0.021 | 0.081 ± 0.000 | 0.347 ± 0.003 | 0.053 ± 0.307 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | 0.918 ± 0.010 | 0.044 ± 0.000 | 0.210 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | 0.922 ± 0.018 | 0.046 ± 0.000 | 0.214 ± 0.003 | 0.022 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | 0.189 ± 0.002 | 0.050 ± 0.000 | 0.175 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | 0.173 ± 0.003 | 0.040 ± 0.000 | 0.160 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | 0.235 ± 0.004 | 0.079 ± 0.000 | 0.223 ± 0.005 | 0.122 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | 0.175 ± 0.004 | 0.041 ± 0.000 | 0.159 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | 0.178 ± 0.002 | 0.043 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | 1.328 ± 0.028 | 0.079 ± 0.000 | 0.164 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | 1.258 ± 0.020 | 0.056 ± 0.000 | 0.145 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | 1.401 ± 0.027 | 0.141 ± 0.001 | 0.218 ± 0.004 | 0.076 ± 0.018 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | 1.309 ± 0.030 | 0.057 ± 0.000 | 0.145 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | 1.318 ± 0.013 | 0.062 ± 0.000 | 0.149 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | 0.168 ± 0.002 | 0.077 ± 0.000 | 0.166 ± 0.005 | 0.018 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | 0.149 ± 0.002 | 0.055 ± 0.000 | 0.147 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | 0.230 ± 0.004 | 0.139 ± 0.001 | 0.236 ± 0.004 | 0.091 ± 0.123 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | 0.150 ± 0.003 | 0.056 ± 0.000 | 0.147 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | 0.153 ± 0.002 | 0.061 ± 0.000 | 0.149 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | 0.175 ± 0.002 | 0.040 ± 0.000 | 0.334 ± 0.001 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | 0.161 ± 0.005 | 0.034 ± 0.000 | 0.325 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | 0.188 ± 0.003 | 0.050 ± 0.000 | 0.354 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | 0.162 ± 0.003 | 0.034 ± 0.000 | 0.325 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | 0.166 ± 0.005 | 0.036 ± 0.000 | 0.327 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | 0.074 ± 0.001 | 0.039 ± 0.000 | 0.144 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | 0.068 ± 0.001 | 0.034 ± 0.000 | 0.138 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | 0.082 ± 0.002 | 0.049 ± 0.000 | 0.164 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | 0.069 ± 0.001 | 0.034 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | 0.072 ± 0.002 | 0.036 ± 0.000 | 0.139 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | 0.321 ± 0.002 | 0.102 ± 0.000 | 0.257 ± 0.004 | 0.069 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | 0.193 ± 0.004 | 0.025 ± 0.000 | 0.185 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | 0.645 ± 0.012 | 0.314 ± 0.001 | 0.453 ± 0.003 | 0.204 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | 0.208 ± 0.005 | 0.030 ± 0.000 | 0.191 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | 0.231 ± 0.002 | 0.046 ± 0.000 | 0.208 ± 0.004 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | 0.118 ± 0.002 | 0.100 ± 0.000 | 0.174 ± 0.004 | 0.069 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | 0.061 ± 0.001 | 0.023 ± 0.000 | 0.111 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | 0.278 ± 0.003 | 0.313 ± 0.001 | 0.351 ± 0.009 | 0.204 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | 0.064 ± 0.001 | 0.029 ± 0.000 | 0.118 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | 0.075 ± 0.001 | 0.044 ± 0.000 | 0.130 ± 0.001 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

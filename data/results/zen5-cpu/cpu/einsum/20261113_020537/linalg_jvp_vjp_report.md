# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_020537/cpu_ops_t1_20261113_020537.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 713.047 ± 0.956 | 496.973 ± 2.580 | 634.739 ± 3.511 | 491.925 ± 1.138 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 21.994 ± 0.075 | 16.280 ± 0.026 | 14.061 ± 0.037 | 11.671 ± 0.022 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 115.946 ± 0.280 | 83.762 ± 0.218 | 94.204 ± 0.425 | 74.578 ± 1.032 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 428.316 ± 0.786 | 498.077 ± 2.209 | 459.244 ± 3.721 | 483.056 ± 2.886 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 12.140 ± 0.065 | 16.323 ± 0.033 | 11.388 ± 0.042 | 11.713 ± 0.043 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 67.665 ± 0.250 | 83.711 ± 0.202 | 72.911 ± 0.365 | 73.564 ± 0.205 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 738.543 ± 1.107 | 351.497 ± 0.456 | 778.889 ± 1.588 | 354.277 ± 9.896 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 13.583 ± 0.081 | 6.552 ± 0.023 | 13.914 ± 0.044 | 6.825 ± 0.181 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 93.044 ± 0.168 | 46.259 ± 0.149 | 113.892 ± 0.241 | 48.460 ± 0.163 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 725.783 ± 1.871 | 360.663 ± 1.507 | 407.188 ± 0.960 | 348.968 ± 0.821 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 12.230 ± 0.073 | 6.925 ± 0.022 | 7.460 ± 0.105 | 6.452 ± 0.011 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 90.845 ± 0.078 | 49.441 ± 0.089 | 60.853 ± 0.543 | 46.881 ± 0.086 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 993.358 ± 7.614 | 543.030 ± 1.127 | 530.823 ± 2.216 | 495.998 ± 4.838 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 19.070 ± 0.044 | 10.516 ± 0.026 | 10.527 ± 0.024 | 9.268 ± 0.019 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 129.292 ± 0.254 | 73.858 ± 0.245 | 82.346 ± 0.412 | 67.401 ± 0.110 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 857.598 ± 2.753 | 541.347 ± 0.861 | 514.695 ± 1.740 | 498.209 ± 5.790 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 14.528 ± 0.110 | 10.535 ± 0.028 | 10.620 ± 0.049 | 9.174 ± 0.015 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 108.075 ± 0.230 | 73.386 ± 0.094 | 86.139 ± 0.428 | 68.285 ± 0.276 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 58.003 ± 0.134 | 56.732 ± 0.218 | 49.388 ± 0.242 | 41.448 ± 0.085 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.577 ± 0.035 | 1.294 ± 0.002 | 1.409 ± 0.032 | 0.977 ± 0.009 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 8.935 ± 0.109 | 7.980 ± 0.017 | 7.453 ± 0.245 | 6.568 ± 0.015 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 30.034 ± 0.146 | 56.136 ± 0.104 | 85.667 ± 4.050 | 40.276 ± 0.098 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.837 ± 0.031 | 1.296 ± 0.013 | 2.055 ± 0.038 | 0.969 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.628 ± 0.130 | 7.913 ± 0.009 | 13.234 ± 0.123 | 6.612 ± 0.022 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 1537.732 ± 14.203 | 1096.173 ± 3.909 | 1154.158 ± 4.239 | 904.484 ± 1.715 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 39.084 ± 0.144 | 28.660 ± 0.040 | 25.814 ± 0.114 | 20.602 ± 0.106 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 228.792 ± 1.036 | 168.364 ± 0.594 | 196.733 ± 2.738 | 143.114 ± 0.377 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 831.474 ± 16.232 | 1119.245 ± 1.448 | 870.117 ± 2.726 | 910.452 ± 2.690 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 20.406 ± 0.039 | 28.646 ± 0.060 | 21.552 ± 0.075 | 20.587 ± 0.040 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 122.716 ± 0.310 | 168.106 ± 0.442 | 157.251 ± 0.469 | 142.319 ± 0.290 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.297 ± 0.004 | 0.081 ± 0.000 | 0.158 ± 0.005 | 0.053 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.227 ± 0.002 | 0.028 ± 0.000 | 0.121 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.485 ± 0.004 | 0.240 ± 0.000 | 0.252 ± 0.006 | 0.141 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.228 ± 0.004 | 0.032 ± 0.000 | 0.124 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.243 ± 0.003 | 0.043 ± 0.000 | 0.132 ± 0.002 | 0.024 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.089 ± 0.001 | 0.081 ± 0.000 | 0.134 ± 0.002 | 0.053 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.057 ± 0.000 | 0.027 ± 0.000 | 0.102 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.185 ± 0.001 | 0.238 ± 0.001 | 0.216 ± 0.005 | 0.142 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.060 ± 0.000 | 0.031 ± 0.000 | 0.105 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.066 ± 0.001 | 0.042 ± 0.000 | 0.111 ± 0.002 | 0.026 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.911 ± 0.008 | 0.045 ± 0.000 | 0.220 ± 0.006 | 0.026 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.873 ± 0.017 | 0.036 ± 0.000 | 0.199 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.959 ± 0.014 | 0.071 ± 0.000 | 0.274 ± 0.005 | 0.051 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.885 ± 0.011 | 0.037 ± 0.000 | 0.196 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.894 ± 0.009 | 0.039 ± 0.000 | 0.201 ± 0.005 | 0.021 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.166 ± 0.001 | 0.044 ± 0.000 | 0.172 ± 0.005 | 0.027 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.152 ± 0.002 | 0.034 ± 0.000 | 0.154 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.208 ± 0.002 | 0.071 ± 0.000 | 0.211 ± 0.005 | 0.051 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.153 ± 0.002 | 0.034 ± 0.000 | 0.153 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.156 ± 0.001 | 0.037 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 1.289 ± 0.034 | 0.069 ± 0.000 | 0.158 ± 0.006 | 0.030 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 1.225 ± 0.010 | 0.046 ± 0.000 | 0.139 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 1.392 ± 0.006 | 0.130 ± 0.000 | 0.199 ± 0.005 | 0.062 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 1.268 ± 0.027 | 0.048 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 1.285 ± 0.005 | 0.052 ± 0.000 | 0.143 ± 0.003 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.149 ± 0.001 | 0.068 ± 0.000 | 0.164 ± 0.006 | 0.030 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.129 ± 0.002 | 0.046 ± 0.000 | 0.142 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.206 ± 0.002 | 0.130 ± 0.000 | 0.202 ± 0.006 | 0.061 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.130 ± 0.001 | 0.047 ± 0.000 | 0.144 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.134 ± 0.001 | 0.051 ± 0.000 | 0.145 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.166 ± 0.004 | 0.034 ± 0.000 | 0.332 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.156 ± 0.003 | 0.028 ± 0.000 | 0.322 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.177 ± 0.001 | 0.044 ± 0.000 | 0.350 ± 0.006 | 0.029 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.154 ± 0.002 | 0.029 ± 0.000 | 0.323 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.158 ± 0.003 | 0.030 ± 0.000 | 0.325 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.067 ± 0.001 | 0.034 ± 0.000 | 0.145 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.062 ± 0.000 | 0.028 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.075 ± 0.001 | 0.044 ± 0.000 | 0.164 ± 0.004 | 0.033 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.063 ± 0.001 | 0.029 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.064 ± 0.001 | 0.030 ± 0.000 | 0.140 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.312 ± 0.004 | 0.097 ± 0.000 | 0.259 ± 0.008 | 0.078 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.188 ± 0.002 | 0.020 ± 0.000 | 0.183 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.630 ± 0.004 | 0.308 ± 0.000 | 0.432 ± 0.003 | 0.210 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.201 ± 0.004 | 0.026 ± 0.000 | 0.188 ± 0.006 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.219 ± 0.001 | 0.041 ± 0.000 | 0.208 ± 0.005 | 0.033 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.113 ± 0.001 | 0.096 ± 0.000 | 0.175 ± 0.006 | 0.079 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.055 ± 0.001 | 0.020 ± 0.000 | 0.112 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.269 ± 0.001 | 0.308 ± 0.000 | 0.338 ± 0.007 | 0.210 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.059 ± 0.001 | 0.025 ± 0.000 | 0.117 ± 0.002 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.070 ± 0.001 | 0.041 ± 0.000 | 0.133 ± 0.001 | 0.034 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_031144/cpu_ops_t4_20261113_031144.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 264.629 ± 0.985 | 212.834 ± 0.510 | 201.070 ± 1.065 | 252.555 ± 4.349 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 20.389 ± 0.063 | 15.247 ± 0.036 | 6.026 ± 0.030 | 11.294 ± 0.669 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 53.204 ± 0.194 | 45.632 ± 1.082 | 33.649 ± 0.170 | 52.137 ± 4.026 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 150.893 ± 0.325 | 212.407 ± 0.761 | 140.443 ± 2.735 | 256.938 ± 9.147 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.416 ± 0.032 | 15.265 ± 0.010 | 5.231 ± 0.021 | 11.113 ± 0.458 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.361 ± 0.223 | 45.188 ± 0.428 | 25.244 ± 0.094 | 50.360 ± 2.551 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 214.406 ± 0.982 | 104.323 ± 0.686 | 247.948 ± 1.427 | 180.203 ± 9.748 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.794 ± 0.072 | 2.886 ± 0.016 | 5.552 ± 0.100 | 5.586 ± 1.264 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 29.778 ± 0.105 | 16.220 ± 0.360 | 42.452 ± 0.494 | 29.706 ± 3.822 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 199.116 ± 0.377 | 107.345 ± 0.293 | 116.522 ± 0.672 | 175.402 ± 7.990 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 4.401 ± 0.056 | 2.981 ± 0.013 | 2.685 ± 0.029 | 5.260 ± 0.676 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 27.230 ± 0.113 | 16.692 ± 0.173 | 18.293 ± 0.127 | 29.629 ± 8.474 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 279.670 ± 0.612 | 159.958 ± 0.396 | 161.664 ± 0.341 | 262.896 ± 7.583 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.913 ± 0.029 | 4.235 ± 0.026 | 4.259 ± 0.063 | 9.353 ± 1.703 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 40.770 ± 0.173 | 25.083 ± 0.506 | 29.209 ± 0.155 | 44.872 ± 4.594 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 233.671 ± 0.668 | 159.224 ± 4.442 | 158.557 ± 1.405 | 248.021 ± 9.939 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.057 ± 0.016 | 4.230 ± 0.032 | 4.536 ± 0.082 | 9.047 ± 0.865 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 31.903 ± 0.061 | 24.122 ± 0.157 | 32.748 ± 0.224 | 46.782 ± 5.473 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 21.911 ± 0.171 | 20.287 ± 0.035 | 13.409 ± 0.095 | 20.361 ± 0.194 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.158 ± 0.042 | 0.809 ± 0.006 | 0.786 ± 0.030 | 0.756 ± 0.022 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 4.341 ± 0.133 | 3.455 ± 0.057 | 2.758 ± 0.117 | 4.223 ± 0.038 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 11.596 ± 0.106 | 19.585 ± 0.066 | 22.972 ± 0.254 | 19.702 ± 0.218 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.605 ± 0.016 | 0.818 ± 0.006 | 0.905 ± 0.036 | 0.757 ± 0.047 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.346 ± 0.088 | 3.476 ± 0.049 | 4.162 ± 0.014 | 4.265 ± 0.066 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 567.500 ± 11.754 | 406.784 ± 2.792 | 430.255 ± 1.087 | 522.954 ± 8.597 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 37.516 ± 0.071 | 27.124 ± 0.078 | 16.492 ± 0.034 | 19.029 ± 0.308 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 95.203 ± 0.301 | 77.584 ± 0.267 | 84.846 ± 0.230 | 108.444 ± 3.493 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 295.189 ± 0.525 | 429.216 ± 0.861 | 350.452 ± 1.016 | 531.103 ± 9.847 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 18.817 ± 0.075 | 27.136 ± 0.040 | 15.334 ± 0.047 | 19.001 ± 0.235 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 50.124 ± 0.209 | 77.901 ± 0.283 | 78.074 ± 1.864 | 108.108 ± 3.027 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.309 ± 0.004 | 0.086 ± 0.000 | 0.158 ± 0.005 | 0.042 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.234 ± 0.003 | 0.033 ± 0.000 | 0.122 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.501 ± 0.003 | 0.243 ± 0.001 | 0.259 ± 0.005 | 0.133 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.245 ± 0.002 | 0.037 ± 0.000 | 0.124 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.263 ± 0.002 | 0.048 ± 0.000 | 0.132 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.096 ± 0.001 | 0.085 ± 0.000 | 0.135 ± 0.001 | 0.043 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.063 ± 0.000 | 0.032 ± 0.000 | 0.105 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.192 ± 0.001 | 0.238 ± 0.000 | 0.225 ± 0.006 | 0.134 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.066 ± 0.000 | 0.036 ± 0.000 | 0.105 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.072 ± 0.000 | 0.047 ± 0.000 | 0.111 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.931 ± 0.011 | 0.051 ± 0.000 | 0.230 ± 0.008 | 0.020 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.900 ± 0.017 | 0.042 ± 0.000 | 0.206 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.984 ± 0.010 | 0.078 ± 0.000 | 0.281 ± 0.007 | 0.053 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.903 ± 0.006 | 0.043 ± 0.000 | 0.204 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.919 ± 0.011 | 0.045 ± 0.000 | 0.209 ± 0.004 | 0.022 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.185 ± 0.002 | 0.048 ± 0.000 | 0.175 ± 0.005 | 0.019 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.169 ± 0.002 | 0.038 ± 0.000 | 0.161 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.228 ± 0.002 | 0.076 ± 0.000 | 0.216 ± 0.005 | 0.044 ± 0.008 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.171 ± 0.001 | 0.039 ± 0.000 | 0.162 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.174 ± 0.002 | 0.041 ± 0.000 | 0.165 ± 0.005 | 0.021 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 1.335 ± 0.005 | 0.077 ± 0.000 | 0.162 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 1.255 ± 0.012 | 0.054 ± 0.000 | 0.142 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 1.398 ± 0.015 | 0.140 ± 0.001 | 0.205 ± 0.004 | 0.069 ± 0.024 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 1.291 ± 0.023 | 0.055 ± 0.000 | 0.142 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 1.299 ± 0.010 | 0.060 ± 0.000 | 0.149 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.164 ± 0.002 | 0.075 ± 0.000 | 0.168 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.146 ± 0.002 | 0.053 ± 0.000 | 0.149 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.225 ± 0.002 | 0.139 ± 0.000 | 0.207 ± 0.006 | 0.069 ± 0.030 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.146 ± 0.001 | 0.055 ± 0.000 | 0.148 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.150 ± 0.002 | 0.059 ± 0.000 | 0.151 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.172 ± 0.005 | 0.039 ± 0.000 | 0.335 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.157 ± 0.004 | 0.033 ± 0.000 | 0.326 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.181 ± 0.004 | 0.050 ± 0.000 | 0.354 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.162 ± 0.005 | 0.033 ± 0.000 | 0.325 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.165 ± 0.002 | 0.035 ± 0.000 | 0.328 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.072 ± 0.001 | 0.039 ± 0.000 | 0.145 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.067 ± 0.001 | 0.033 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.080 ± 0.001 | 0.048 ± 0.000 | 0.165 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.067 ± 0.001 | 0.033 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.069 ± 0.001 | 0.035 ± 0.000 | 0.139 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.321 ± 0.007 | 0.100 ± 0.000 | 0.263 ± 0.006 | 0.068 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.192 ± 0.002 | 0.023 ± 0.000 | 0.185 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.640 ± 0.003 | 0.313 ± 0.001 | 0.450 ± 0.005 | 0.203 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.205 ± 0.003 | 0.029 ± 0.000 | 0.190 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.229 ± 0.003 | 0.044 ± 0.000 | 0.210 ± 0.004 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.117 ± 0.000 | 0.100 ± 0.000 | 0.178 ± 0.005 | 0.069 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.060 ± 0.001 | 0.022 ± 0.000 | 0.113 ± 0.000 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.277 ± 0.003 | 0.312 ± 0.000 | 0.348 ± 0.009 | 0.203 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.063 ± 0.001 | 0.028 ± 0.000 | 0.119 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.074 ± 0.002 | 0.043 ± 0.000 | 0.134 ± 0.001 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

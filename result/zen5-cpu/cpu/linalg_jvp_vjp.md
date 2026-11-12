# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `2ee56deba06eaad06c590db40867fcef00165e86`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_020537/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_020537/cpu_ops_t1_20261113_020537.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

Both tenferro arms return the same requested outputs -- the primal and the
declared derivative -- with fixtures and wrapping outside the clock. They are
not the same execution: the eager sample runs the forward pass in a borrowed
session and then the derivative call, which carries the eager API's own AD
preparation on every call (its VJP reuses a prepared derivative when the cache
holds one, its JVP re-prepares), because that API exposes no separate prepare
step. The trace arm compiles and prepares the program before sampling. So the
trace number is a prepared-program steady state and the eager number is the
per-call cost of the public API, not a like-for-like difference in one
operation.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager per call, incl. AD prepare (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_031144/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_031144/cpu_ops_t4_20261113_031144.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

Both tenferro arms return the same requested outputs -- the primal and the
declared derivative -- with fixtures and wrapping outside the clock. They are
not the same execution: the eager sample runs the forward pass in a borrowed
session and then the derivative call, which carries the eager API's own AD
preparation on every call (its VJP reuses a prepared derivative when the cache
holds one, its JVP re-prepares), because that API exposes no separate prepare
step. The trace arm compiles and prepares the program before sampling. So the
trace number is a prepared-program steady state and the eager number is the
per-call cost of the public API, not a like-for-like difference in one
operation.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager per call, incl. AD prepare (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_035559/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_035559/cpu_ops_t8_20261113_035559.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

Both tenferro arms return the same requested outputs -- the primal and the
declared derivative -- with fixtures and wrapping outside the clock. They are
not the same execution: the eager sample runs the forward pass in a borrowed
session and then the derivative call, which carries the eager API's own AD
preparation on every call (its VJP reuses a prepared derivative when the cache
holds one, its JVP re-prepares), because that API exposes no separate prepare
step. The trace arm compiles and prepares the program before sampling. So the
trace number is a prepared-program steady state and the eager number is the
per-call cost of the public API, not a like-for-like difference in one
operation.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager per call, incl. AD prepare (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
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

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_131021/cpu_ops_t4_20261124_131021.csv`

## Linalg JVP/VJP Benchmark Items

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
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 390.692 ± 2.732 | 212.474 ± 1.725 | 205.858 ± 3.449 | 266.314 ± 15.836 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 30.068 ± 0.138 | 15.225 ± 0.042 | 6.117 ± 0.333 | 11.278 ± 0.404 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 77.041 ± 0.387 | 45.155 ± 0.152 | 33.165 ± 0.170 | 52.541 ± 6.444 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 263.089 ± 0.879 | 212.337 ± 1.185 | 139.981 ± 0.787 | 257.894 ± 8.933 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 20.050 ± 0.049 | 15.260 ± 0.069 | 5.259 ± 0.056 | 10.561 ± 0.918 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 52.730 ± 0.353 | 45.190 ± 0.396 | 25.870 ± 0.309 | 50.038 ± 3.555 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 236.071 ± 1.844 | 104.080 ± 0.440 | 237.293 ± 2.212 | 195.873 ± 20.014 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 6.506 ± 0.072 | 2.921 ± 0.037 | 5.634 ± 0.101 | 5.896 ± 1.358 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 32.763 ± 0.233 | 15.894 ± 0.093 | 42.166 ± 0.833 | 30.253 ± 3.541 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 216.169 ± 0.657 | 107.250 ± 0.724 | 119.023 ± 1.986 | 176.915 ± 6.844 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 5.103 ± 0.117 | 3.016 ± 0.050 | 2.702 ± 0.027 | 5.129 ± 3.361 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 30.289 ± 0.201 | 16.773 ± 0.347 | 18.294 ± 0.112 | 30.458 ± 7.328 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 328.310 ± 3.801 | 159.664 ± 1.490 | 165.909 ± 1.959 | 274.140 ± 28.374 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 9.429 ± 0.058 | 4.265 ± 0.036 | 4.273 ± 0.120 | 8.967 ± 0.885 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 48.576 ± 0.343 | 24.458 ± 0.389 | 28.271 ± 0.171 | 43.726 ± 4.475 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 280.293 ± 0.550 | 159.462 ± 0.658 | 166.959 ± 2.489 | 247.898 ± 10.270 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 6.585 ± 0.304 | 4.261 ± 0.014 | 4.575 ± 0.104 | 9.064 ± 0.973 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 39.576 ± 0.912 | 24.310 ± 0.335 | 32.662 ± 0.638 | 46.715 ± 7.096 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 31.496 ± 0.208 | 20.322 ± 0.151 | 12.965 ± 0.125 | 20.391 ± 0.324 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.499 ± 0.029 | 0.839 ± 0.015 | 1.501 ± 0.054 | 0.750 ± 0.017 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.964 ± 0.039 | 3.417 ± 0.073 | 2.894 ± 0.273 | 4.224 ± 0.785 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 20.817 ± 0.306 | 19.380 ± 0.254 | 22.690 ± 0.625 | 19.405 ± 0.396 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.982 ± 0.049 | 0.831 ± 0.014 | 0.891 ± 0.028 | 0.787 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.818 ± 0.157 | 3.451 ± 0.036 | 4.340 ± 0.086 | 4.291 ± 0.068 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 835.165 ± 4.299 | 408.223 ± 5.566 | 434.952 ± 6.654 | 528.512 ± 9.372 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 55.795 ± 0.339 | 27.317 ± 0.251 | 16.671 ± 0.246 | 19.061 ± 0.568 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 140.747 ± 0.946 | 77.956 ± 0.178 | 89.656 ± 1.710 | 109.057 ± 3.049 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 557.170 ± 8.789 | 431.547 ± 3.515 | 353.103 ± 1.849 | 545.535 ± 18.525 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 36.898 ± 0.056 | 27.297 ± 0.079 | 15.406 ± 0.149 | 19.041 ± 0.279 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 95.165 ± 0.453 | 77.926 ± 0.502 | 84.260 ± 2.054 | 110.073 ± 2.796 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.355 ± 0.007 | 0.088 ± 0.000 | 0.159 ± 0.002 | 0.043 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.249 ± 0.007 | 0.034 ± 0.000 | 0.121 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.642 ± 0.012 | 0.243 ± 0.002 | 0.261 ± 0.007 | 0.131 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.258 ± 0.003 | 0.038 ± 0.000 | 0.126 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.282 ± 0.003 | 0.049 ± 0.001 | 0.135 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.143 ± 0.001 | 0.086 ± 0.001 | 0.135 ± 0.003 | 0.043 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.079 ± 0.001 | 0.032 ± 0.000 | 0.106 ± 0.010 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.328 ± 0.004 | 0.241 ± 0.001 | 0.228 ± 0.008 | 0.132 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.084 ± 0.002 | 0.036 ± 0.000 | 0.110 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.097 ± 0.001 | 0.048 ± 0.000 | 0.116 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.964 ± 0.016 | 0.053 ± 0.000 | 0.228 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.928 ± 0.010 | 0.044 ± 0.000 | 0.208 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 1.028 ± 0.021 | 0.080 ± 0.000 | 0.285 ± 0.005 | 0.059 ± 0.028 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.948 ± 0.047 | 0.044 ± 0.000 | 0.203 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.954 ± 0.014 | 0.046 ± 0.000 | 0.212 ± 0.010 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.216 ± 0.001 | 0.050 ± 0.001 | 0.179 ± 0.008 | 0.019 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.196 ± 0.002 | 0.039 ± 0.000 | 0.164 ± 0.009 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.265 ± 0.005 | 0.078 ± 0.001 | 0.220 ± 0.062 | 0.060 ± 0.051 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.199 ± 0.005 | 0.040 ± 0.000 | 0.162 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.272 ± 0.063 | 0.042 ± 0.000 | 0.166 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 1.367 ± 0.016 | 0.078 ± 0.000 | 0.160 ± 0.006 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 1.277 ± 0.042 | 0.055 ± 0.001 | 0.144 ± 0.020 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 1.442 ± 0.008 | 0.143 ± 0.001 | 0.208 ± 0.007 | 0.122 ± 0.053 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 1.301 ± 0.019 | 0.056 ± 0.001 | 0.143 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 1.322 ± 0.027 | 0.061 ± 0.000 | 0.148 ± 0.003 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.205 ± 0.001 | 0.077 ± 0.001 | 0.168 ± 0.007 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.175 ± 0.003 | 0.054 ± 0.000 | 0.146 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.293 ± 0.004 | 0.141 ± 0.001 | 0.207 ± 0.005 | 0.063 ± 0.017 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.179 ± 0.004 | 0.055 ± 0.000 | 0.149 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.183 ± 0.004 | 0.059 ± 0.001 | 0.155 ± 0.008 | 0.017 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.194 ± 0.003 | 0.039 ± 0.001 | 0.340 ± 0.010 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.185 ± 0.004 | 0.034 ± 0.000 | 0.332 ± 0.016 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.215 ± 0.003 | 0.050 ± 0.000 | 0.355 ± 0.009 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.186 ± 0.004 | 0.034 ± 0.000 | 0.334 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.189 ± 0.003 | 0.035 ± 0.000 | 0.330 ± 0.008 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.098 ± 0.001 | 0.038 ± 0.000 | 0.145 ± 0.003 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.091 ± 0.002 | 0.033 ± 0.000 | 0.140 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.110 ± 0.002 | 0.049 ± 0.000 | 0.165 ± 0.006 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.093 ± 0.002 | 0.033 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.094 ± 0.002 | 0.035 ± 0.000 | 0.141 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.396 ± 0.005 | 0.101 ± 0.000 | 0.265 ± 0.016 | 0.069 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.216 ± 0.005 | 0.024 ± 0.000 | 0.185 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.876 ± 0.012 | 0.313 ± 0.001 | 0.451 ± 0.012 | 0.203 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.226 ± 0.002 | 0.030 ± 0.000 | 0.195 ± 0.009 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.262 ± 0.005 | 0.045 ± 0.000 | 0.215 ± 0.010 | 0.022 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.193 ± 0.001 | 0.100 ± 0.001 | 0.179 ± 0.005 | 0.069 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.080 ± 0.004 | 0.023 ± 0.000 | 0.113 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.505 ± 0.003 | 0.313 ± 0.001 | 0.349 ± 0.013 | 0.205 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.089 ± 0.002 | 0.029 ± 0.000 | 0.120 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.001 | 0.044 ± 0.000 | 0.136 ± 0.003 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

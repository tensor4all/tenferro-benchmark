# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

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

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_135202/cpu_ops_t8_20261124_135202.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | 288.497 ± 1.088 | 172.476 ± 0.369 | 132.890 ± 1.029 | 209.028 ± 19.220 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | 30.314 ± 0.178 | 15.152 ± 0.068 | 4.797 ± 0.128 | 11.566 ± 1.861 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | 67.781 ± 0.410 | 41.459 ± 0.076 | 22.913 ± 0.881 | 47.599 ± 8.322 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | 200.425 ± 1.229 | 172.145 ± 0.112 | 95.517 ± 0.671 | 199.308 ± 8.888 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | 20.128 ± 0.079 | 15.178 ± 0.023 | 4.335 ± 0.132 | 11.578 ± 1.745 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | 45.862 ± 0.113 | 41.390 ± 0.170 | 17.907 ± 0.093 | 45.443 ± 4.763 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | 153.869 ± 0.765 | 64.606 ± 0.295 | 148.207 ± 1.491 | 125.824 ± 17.963 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | 5.878 ± 0.068 | 2.451 ± 0.012 | 4.054 ± 0.057 | 6.156 ± 2.999 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | 22.626 ± 0.121 | 11.172 ± 0.019 | 27.153 ± 0.287 | 26.547 ± 6.721 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | 127.946 ± 2.005 | 66.608 ± 0.375 | 71.698 ± 0.508 | 128.366 ± 10.394 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | 4.205 ± 0.076 | 2.526 ± 0.006 | 2.082 ± 0.016 | 5.405 ± 1.228 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | 20.203 ± 0.086 | 11.638 ± 0.012 | 11.469 ± 0.141 | 23.060 ± 8.901 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | 197.271 ± 4.837 | 95.086 ± 1.032 | 111.006 ± 1.639 | 181.262 ± 16.016 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | 8.615 ± 0.151 | 3.522 ± 0.018 | 3.429 ± 0.010 | 9.633 ± 1.509 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | 31.977 ± 0.105 | 15.839 ± 0.094 | 19.869 ± 0.150 | 40.602 ± 11.086 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | 159.971 ± 0.906 | 95.178 ± 0.496 | 104.357 ± 0.858 | 183.955 ± 9.599 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | 5.246 ± 0.036 | 3.490 ± 0.017 | 3.647 ± 0.026 | 10.402 ± 2.731 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | 25.416 ± 0.071 | 15.676 ± 0.017 | 21.034 ± 0.424 | 42.565 ± 2.283 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | 21.917 ± 0.549 | 13.913 ± 0.134 | 9.341 ± 0.186 | 14.836 ± 1.520 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | 1.577 ± 0.104 | 0.773 ± 0.005 | 0.783 ± 0.042 | 0.578 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | 5.172 ± 0.173 | 2.976 ± 0.019 | 2.365 ± 0.038 | 3.671 ± 0.051 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | 15.095 ± 0.543 | 13.500 ± 0.097 | 15.233 ± 0.135 | 14.293 ± 1.864 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | 0.986 ± 0.016 | 0.776 ± 0.003 | 0.891 ± 0.032 | 0.589 ± 0.014 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | 3.465 ± 0.082 | 2.905 ± 0.015 | 3.306 ± 0.021 | 3.635 ± 0.095 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | 625.106 ± 1.799 | 301.961 ± 1.319 | 326.793 ± 5.452 | 460.735 ± 18.142 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | 56.651 ± 0.057 | 27.517 ± 0.244 | 15.135 ± 0.311 | 19.984 ± 1.628 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | 119.545 ± 0.233 | 68.119 ± 0.229 | 68.777 ± 0.407 | 106.513 ± 4.361 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | 407.893 ± 1.425 | 324.704 ± 0.448 | 275.797 ± 6.173 | 462.670 ± 13.096 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | 37.699 ± 0.327 | 27.376 ± 0.026 | 14.053 ± 0.041 | 19.791 ± 1.489 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | 80.657 ± 0.096 | 68.343 ± 0.366 | 63.918 ± 0.308 | 102.049 ± 4.467 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | 0.361 ± 0.004 | 0.089 ± 0.000 | 0.160 ± 0.005 | 0.042 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | 0.255 ± 0.002 | 0.035 ± 0.000 | 0.124 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | 0.646 ± 0.006 | 0.244 ± 0.001 | 0.259 ± 0.006 | 0.131 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | 0.266 ± 0.001 | 0.039 ± 0.000 | 0.126 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | 0.288 ± 0.004 | 0.051 ± 0.000 | 0.133 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | 0.146 ± 0.002 | 0.088 ± 0.000 | 0.135 ± 0.001 | 0.043 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | 0.082 ± 0.001 | 0.034 ± 0.000 | 0.104 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | 0.329 ± 0.002 | 0.242 ± 0.000 | 0.224 ± 0.006 | 0.133 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | 0.087 ± 0.001 | 0.038 ± 0.000 | 0.107 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | 0.100 ± 0.001 | 0.049 ± 0.000 | 0.112 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | 0.982 ± 0.007 | 0.055 ± 0.000 | 0.230 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | 0.928 ± 0.005 | 0.045 ± 0.000 | 0.213 ± 0.010 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | 1.030 ± 0.019 | 0.082 ± 0.000 | 0.338 ± 0.008 | 0.248 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | 0.953 ± 0.008 | 0.046 ± 0.000 | 0.209 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | 0.970 ± 0.012 | 0.048 ± 0.000 | 0.211 ± 0.004 | 0.022 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | 0.218 ± 0.002 | 0.052 ± 0.000 | 0.180 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | 0.201 ± 0.003 | 0.041 ± 0.000 | 0.164 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | 0.269 ± 0.004 | 0.080 ± 0.000 | 0.223 ± 0.005 | 0.060 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | 0.200 ± 0.001 | 0.042 ± 0.000 | 0.164 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | 0.205 ± 0.001 | 0.044 ± 0.000 | 0.166 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | 1.372 ± 0.020 | 0.081 ± 0.000 | 0.164 ± 0.006 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | 1.268 ± 0.008 | 0.057 ± 0.000 | 0.145 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | 1.457 ± 0.034 | 0.145 ± 0.001 | 0.216 ± 0.008 | 0.244 ± 0.226 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | 1.315 ± 0.015 | 0.058 ± 0.000 | 0.144 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | 1.323 ± 0.006 | 0.063 ± 0.000 | 0.149 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | 0.209 ± 0.002 | 0.078 ± 0.000 | 0.166 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | 0.180 ± 0.002 | 0.055 ± 0.000 | 0.149 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | 0.296 ± 0.002 | 0.142 ± 0.001 | 0.234 ± 0.005 | 0.091 ± 0.020 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | 0.182 ± 0.003 | 0.057 ± 0.000 | 0.150 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | 0.188 ± 0.002 | 0.061 ± 0.000 | 0.154 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | 0.202 ± 0.001 | 0.040 ± 0.000 | 0.336 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | 0.187 ± 0.005 | 0.035 ± 0.000 | 0.324 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | 0.222 ± 0.002 | 0.050 ± 0.000 | 0.351 ± 0.008 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | 0.190 ± 0.003 | 0.035 ± 0.000 | 0.327 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | 0.194 ± 0.002 | 0.036 ± 0.000 | 0.328 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | 0.101 ± 0.001 | 0.039 ± 0.000 | 0.146 ± 0.002 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | 0.094 ± 0.001 | 0.034 ± 0.000 | 0.139 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | 0.116 ± 0.001 | 0.050 ± 0.000 | 0.168 ± 0.007 | 0.020 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | 0.097 ± 0.004 | 0.034 ± 0.000 | 0.139 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | 0.096 ± 0.001 | 0.036 ± 0.000 | 0.142 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | 0.399 ± 0.002 | 0.103 ± 0.000 | 0.260 ± 0.005 | 0.069 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | 0.219 ± 0.003 | 0.025 ± 0.000 | 0.185 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | 0.876 ± 0.003 | 0.314 ± 0.001 | 0.450 ± 0.007 | 0.201 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | 0.232 ± 0.003 | 0.031 ± 0.000 | 0.191 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | 0.267 ± 0.002 | 0.046 ± 0.000 | 0.211 ± 0.004 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | 0.193 ± 0.002 | 0.101 ± 0.000 | 0.176 ± 0.005 | 0.070 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | 0.082 ± 0.002 | 0.024 ± 0.000 | 0.114 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | 0.503 ± 0.003 | 0.314 ± 0.001 | 0.356 ± 0.014 | 0.204 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | 0.089 ± 0.002 | 0.029 ± 0.000 | 0.118 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | 0.110 ± 0.002 | 0.045 ± 0.000 | 0.133 ± 0.001 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

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

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_120608/cpu_ops_t1_20261124_120608.csv`

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
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 998.446 ± 0.895 | 498.099 ± 1.639 | 635.665 ± 8.507 | 484.899 ± 1.870 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 31.558 ± 0.195 | 16.283 ± 0.096 | 14.149 ± 0.114 | 11.705 ± 0.047 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 165.664 ± 0.736 | 84.527 ± 1.229 | 95.196 ± 0.632 | 76.998 ± 0.666 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 713.765 ± 1.320 | 496.957 ± 2.687 | 457.023 ± 3.657 | 484.523 ± 3.690 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 21.619 ± 0.211 | 16.290 ± 0.045 | 11.496 ± 0.143 | 11.720 ± 0.045 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 116.931 ± 0.690 | 84.397 ± 0.702 | 73.052 ± 0.638 | 75.788 ± 1.533 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 770.984 ± 1.425 | 350.794 ± 1.465 | 776.907 ± 3.882 | 353.168 ± 3.180 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 14.496 ± 0.136 | 6.587 ± 0.019 | 14.012 ± 0.187 | 6.673 ± 0.018 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 98.386 ± 0.211 | 46.414 ± 0.251 | 111.334 ± 0.496 | 47.568 ± 0.392 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 752.504 ± 1.521 | 359.676 ± 1.230 | 407.921 ± 1.919 | 351.747 ± 1.410 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 13.154 ± 0.027 | 6.892 ± 0.009 | 7.477 ± 0.051 | 6.456 ± 0.015 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 95.808 ± 0.314 | 49.293 ± 0.103 | 56.674 ± 0.442 | 46.968 ± 0.236 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 1120.745 ± 4.387 | 538.852 ± 0.795 | 533.443 ± 6.105 | 493.573 ± 0.744 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 22.205 ± 0.184 | 10.506 ± 0.012 | 10.610 ± 0.106 | 9.254 ± 0.097 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 148.927 ± 0.590 | 73.706 ± 0.229 | 84.505 ± 1.535 | 67.579 ± 0.299 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 966.325 ± 1.594 | 539.772 ± 2.344 | 533.453 ± 5.853 | 495.749 ± 5.856 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 17.616 ± 0.177 | 10.556 ± 0.085 | 10.651 ± 0.100 | 9.216 ± 0.029 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 127.674 ± 0.613 | 73.208 ± 0.204 | 84.870 ± 0.435 | 68.296 ± 0.500 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 85.857 ± 0.602 | 56.850 ± 0.129 | 52.967 ± 4.595 | 41.283 ± 0.292 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 2.197 ± 0.016 | 1.313 ± 0.006 | 1.402 ± 0.047 | 0.952 ± 0.008 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 12.828 ± 0.184 | 7.984 ± 0.069 | 7.759 ± 0.058 | 6.564 ± 0.034 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 57.428 ± 0.285 | 56.331 ± 0.237 | 90.104 ± 4.581 | 40.137 ± 0.202 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.438 ± 0.036 | 1.309 ± 0.008 | 2.080 ± 0.043 | 0.966 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 8.447 ± 0.098 | 7.890 ± 0.043 | 13.135 ± 0.085 | 6.621 ± 0.017 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 2230.149 ± 5.820 | 1089.630 ± 2.375 | 1153.383 ± 5.588 | 911.048 ± 5.199 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 57.237 ± 0.717 | 28.800 ± 0.133 | 25.790 ± 0.070 | 20.596 ± 0.076 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 331.189 ± 2.112 | 167.695 ± 0.798 | 189.945 ± 1.542 | 145.171 ± 0.922 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 1515.028 ± 3.219 | 1116.067 ± 4.009 | 871.767 ± 4.545 | 912.195 ± 3.500 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 38.637 ± 0.140 | 28.668 ± 0.111 | 21.684 ± 0.279 | 20.638 ± 0.076 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 227.172 ± 0.494 | 168.524 ± 1.756 | 164.125 ± 3.678 | 144.409 ± 0.973 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.351 ± 0.004 | 0.082 ± 0.001 | 0.158 ± 0.009 | 0.052 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.251 ± 0.004 | 0.028 ± 0.000 | 0.122 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.630 ± 0.016 | 0.235 ± 0.001 | 0.251 ± 0.005 | 0.141 ± 0.002 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.260 ± 0.005 | 0.032 ± 0.000 | 0.124 ± 0.046 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.280 ± 0.006 | 0.043 ± 0.000 | 0.131 ± 0.002 | 0.024 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.136 ± 0.005 | 0.081 ± 0.000 | 0.135 ± 0.002 | 0.052 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.073 ± 0.001 | 0.027 ± 0.001 | 0.104 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.318 ± 0.003 | 0.236 ± 0.001 | 0.221 ± 0.006 | 0.142 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.077 ± 0.003 | 0.031 ± 0.000 | 0.108 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.090 ± 0.002 | 0.043 ± 0.001 | 0.113 ± 0.003 | 0.026 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.957 ± 0.015 | 0.046 ± 0.000 | 0.219 ± 0.006 | 0.026 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.917 ± 0.017 | 0.037 ± 0.000 | 0.197 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 1.010 ± 0.033 | 0.073 ± 0.001 | 0.274 ± 0.003 | 0.051 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.912 ± 0.018 | 0.037 ± 0.006 | 0.195 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.934 ± 0.011 | 0.040 ± 0.000 | 0.198 ± 0.005 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.196 ± 0.002 | 0.044 ± 0.000 | 0.173 ± 0.007 | 0.026 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.178 ± 0.002 | 0.034 ± 0.001 | 0.158 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.247 ± 0.006 | 0.072 ± 0.000 | 0.215 ± 0.008 | 0.051 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.177 ± 0.002 | 0.034 ± 0.001 | 0.160 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.182 ± 0.003 | 0.036 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 1.378 ± 0.046 | 0.069 ± 0.001 | 0.157 ± 0.011 | 0.030 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 1.237 ± 0.036 | 0.046 ± 0.000 | 0.139 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 1.451 ± 0.030 | 0.133 ± 0.004 | 0.201 ± 0.007 | 0.062 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 1.293 ± 0.035 | 0.047 ± 0.000 | 0.139 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 1.308 ± 0.043 | 0.052 ± 0.000 | 0.143 ± 0.003 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.189 ± 0.005 | 0.067 ± 0.001 | 0.164 ± 0.007 | 0.030 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.156 ± 0.003 | 0.045 ± 0.000 | 0.145 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.271 ± 0.005 | 0.130 ± 0.001 | 0.204 ± 0.004 | 0.061 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.160 ± 0.004 | 0.046 ± 0.000 | 0.147 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.165 ± 0.003 | 0.050 ± 0.000 | 0.149 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.194 ± 0.004 | 0.034 ± 0.000 | 0.337 ± 0.007 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.181 ± 0.006 | 0.028 ± 0.000 | 0.321 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.213 ± 0.005 | 0.044 ± 0.000 | 0.352 ± 0.004 | 0.029 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.182 ± 0.004 | 0.029 ± 0.000 | 0.325 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.184 ± 0.003 | 0.030 ± 0.000 | 0.328 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.091 ± 0.001 | 0.033 ± 0.000 | 0.147 ± 0.003 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.085 ± 0.001 | 0.028 ± 0.000 | 0.140 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.103 ± 0.001 | 0.044 ± 0.001 | 0.170 ± 0.010 | 0.033 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.085 ± 0.001 | 0.028 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.088 ± 0.002 | 0.030 ± 0.000 | 0.140 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.396 ± 0.003 | 0.098 ± 0.000 | 0.258 ± 0.003 | 0.077 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.212 ± 0.003 | 0.020 ± 0.000 | 0.182 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.869 ± 0.007 | 0.309 ± 0.001 | 0.442 ± 0.017 | 0.209 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.236 ± 0.019 | 0.026 ± 0.000 | 0.191 ± 0.006 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.262 ± 0.006 | 0.042 ± 0.000 | 0.211 ± 0.007 | 0.033 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.187 ± 0.003 | 0.097 ± 0.000 | 0.178 ± 0.006 | 0.078 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.075 ± 0.001 | 0.020 ± 0.000 | 0.115 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.497 ± 0.013 | 0.310 ± 0.002 | 0.336 ± 0.012 | 0.209 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.084 ± 0.003 | 0.025 ± 0.000 | 0.121 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.104 ± 0.003 | 0.041 ± 0.000 | 0.136 ± 0.005 | 0.033 ± 0.001 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

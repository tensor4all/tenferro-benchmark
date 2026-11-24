# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `bb9869533cc2ad586b71cdb49a3e1f85439cd2b2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_120608/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
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

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_120608/cpu_ops_t1_20261124_120608.csv`

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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_131021/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
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

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_131021/cpu_ops_t4_20261124_131021.csv`

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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_135202/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
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

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_135202/cpu_ops_t8_20261124_135202.csv`

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

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

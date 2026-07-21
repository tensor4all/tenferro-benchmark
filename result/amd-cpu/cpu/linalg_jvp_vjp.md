# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20260721_071519`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20260721_071519`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260721_071519/cpu_ops_t4_20260721_071519.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260721_071519/linalg_jvp_vjp_t4_20260721_071519.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 8.282 ± 0.160 | 8.182 ± 0.193 | 29.690 ± 0.475 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 30.525 ± 0.865 | 28.267 ± 0.312 | 88.789 ± 5.303 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 7.637 ± 0.146 | 6.521 ± 0.412 | 28.563 ± 1.523 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 29.283 ± 0.301 | 21.345 ± 0.344 | 88.646 ± 3.305 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 5.272 ± 0.285 | 4.531 ± 0.157 | 7.324 ± 1.681 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 23.892 ± 0.734 | 24.787 ± 0.528 | 23.799 ± 3.082 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 4.864 ± 0.192 | 2.831 ± 0.135 | 10.714 ± 1.535 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 22.522 ± 0.251 | 11.949 ± 0.296 | 30.937 ± 5.709 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.036 ± 0.282 | 4.608 ± 0.148 | 13.969 ± 0.535 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 30.797 ± 1.169 | 19.190 ± 0.252 | 45.298 ± 2.553 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 6.922 ± 0.567 | 4.363 ± 0.144 | 14.030 ± 0.576 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 28.912 ± 0.887 | 18.894 ± 0.343 | 60.881 ± 2.649 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 1.431 ± 0.081 | 1.591 ± 0.059 | 4.110 ± 0.063 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 5.439 ± 0.142 | 3.263 ± 0.152 | 10.390 ± 0.165 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.943 ± 0.082 | 1.724 ± 0.101 | 5.608 ± 0.187 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 3.246 ± 0.078 | 4.654 ± 0.188 | 11.262 ± 0.227 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.299 ± 0.165 | 18.330 ± 0.410 | 52.039 ± 0.829 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 70.598 ± 0.539 | 81.509 ± 4.477 | 163.776 ± 3.663 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 16.870 ± 0.145 | 16.529 ± 0.290 | 50.427 ± 0.733 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 69.497 ± 0.399 | 64.075 ± 0.696 | 161.626 ± 2.779 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.105 ± 0.021 | 0.338 ± 0.055 | 1.549 ± 0.023 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.125 ± 0.020 | 0.359 ± 0.020 | 1.591 ± 0.125 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.144 ± 0.008 | 0.378 ± 0.012 | 1.801 ± 0.049 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.124 ± 0.018 | 0.325 ± 0.014 | 3.505 ± 0.097 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.132 ± 0.018 | 0.331 ± 0.012 | 3.566 ± 0.358 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.144 ± 0.030 | 0.343 ± 0.019 | 4.753 ± 0.162 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.113 ± 0.016 | 0.523 ± 0.013 | 2.024 ± 0.055 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.111 ± 0.012 | 0.520 ± 0.014 | 1.829 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.131 ± 0.023 | 0.541 ± 0.118 | 3.108 ± 0.051 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.126 ± 0.015 | 0.495 ± 0.049 | 5.214 ± 1.037 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.121 ± 0.007 | 0.485 ± 0.039 | 5.200 ± 0.996 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.128 ± 0.011 | 0.471 ± 0.060 | 5.607 ± 1.994 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.296 ± 0.032 | 0.426 ± 0.030 | 3.184 ± 0.026 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.293 ± 0.014 | 0.416 ± 0.026 | 2.714 ± 0.083 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.292 ± 0.039 | 0.408 ± 0.020 | 2.686 ± 0.073 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.582 ± 0.025 | 0.467 ± 0.048 | 6.398 ± 1.807 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.579 ± 0.014 | 0.432 ± 0.019 | 6.596 ± 1.134 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.579 ± 0.034 | 0.440 ± 0.012 | 6.953 ± 1.089 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.078 ± 0.005 | 0.696 ± 0.019 | 2.348 ± 0.068 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.076 ± 0.006 | 0.697 ± 0.058 | 2.269 ± 0.071 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.071 ± 0.007 | 0.758 ± 0.075 | 2.368 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.103 ± 0.012 | 0.451 ± 0.014 | 4.571 ± 1.193 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.100 ± 0.012 | 0.453 ± 0.011 | 5.176 ± 0.575 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.103 ± 0.003 | 0.491 ± 0.017 | 5.206 ± 0.289 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.069 ± 0.003 | 0.438 ± 0.010 | 1.871 ± 0.064 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.085 ± 0.008 | 0.490 ± 0.052 | 1.884 ± 0.061 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.097 ± 0.011 | 0.466 ± 0.014 | 1.909 ± 0.055 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.087 ± 0.018 | 0.360 ± 0.061 | 4.036 ± 0.937 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.092 ± 0.006 | 0.350 ± 0.013 | 5.083 ± 0.392 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.109 ± 0.013 | 0.374 ± 0.013 | 4.634 ± 0.959 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

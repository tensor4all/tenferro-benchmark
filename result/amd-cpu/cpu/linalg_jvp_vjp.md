# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20260728_023431`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20260728_023431`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/amd-cpu/cpu/einsum/20260728_023431/cpu_ops_t4_20260728_023431.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20260728_023431/linalg_jvp_vjp_t4_20260728_023431.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 66.107 ± 16.591 | 6.816 ± 0.072 | 14.292 ± 4.508 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 174.251 ± 12.171 | 30.463 ± 3.707 | 50.981 ± 9.068 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 55.058 ± 14.406 | 5.805 ± 0.053 | 14.608 ± 3.192 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 97.587 ± 2.734 | 21.206 ± 0.398 | 51.834 ± 12.916 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 32.090 ± 15.525 | 6.614 ± 1.063 | 3.556 ± 0.317 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 91.550 ± 39.987 | 31.477 ± 1.005 | 14.738 ± 2.094 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 32.631 ± 3.838 | 2.935 ± 0.386 | 6.916 ± 0.608 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 184.738 ± 22.504 | 17.800 ± 0.790 | 18.979 ± 4.993 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 43.370 ± 5.736 | 4.160 ± 0.184 | 8.069 ± 1.883 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 197.710 ± 49.207 | 23.685 ± 2.281 | 36.082 ± 5.066 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 41.036 ± 16.969 | 5.393 ± 0.651 | 9.665 ± 1.530 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 225.868 ± 28.018 | 24.974 ± 0.759 | 54.273 ± 9.482 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 11.780 ± 3.452 | 1.233 ± 0.021 | 2.037 ± 0.247 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 72.287 ± 3.812 | 2.977 ± 0.029 | 6.507 ± 0.542 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 12.900 ± 2.603 | 1.514 ± 0.016 | 1.839 ± 0.097 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 84.474 ± 27.820 | 4.569 ± 0.032 | 7.075 ± 0.161 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 96.675 ± 8.376 | 15.876 ± 1.131 | 28.610 ± 10.040 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 197.245 ± 90.302 | 77.337 ± 3.369 | 106.981 ± 10.443 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 91.287 ± 24.502 | 13.555 ± 1.276 | 28.353 ± 9.763 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 186.419 ± 3.642 | 72.322 ± 2.823 | 107.394 ± 13.006 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.230 ± 0.006 | 0.188 ± 0.008 | 0.552 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.236 ± 0.012 | 0.190 ± 0.008 | 0.565 ± 0.015 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.242 ± 0.007 | 0.234 ± 0.006 | 0.563 ± 0.016 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.235 ± 0.008 | 0.183 ± 0.011 | 1.277 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.238 ± 0.010 | 0.187 ± 0.008 | 1.298 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.247 ± 0.011 | 0.229 ± 0.005 | 2.714 ± 1.418 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.221 ± 0.008 | 0.274 ± 0.007 | 0.796 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.226 ± 0.004 | 0.278 ± 0.009 | 0.801 ± 0.021 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.231 ± 0.010 | 0.330 ± 0.014 | 0.817 ± 0.016 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.242 ± 0.007 | 0.264 ± 0.047 | 2.034 ± 0.028 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.244 ± 0.011 | 0.253 ± 0.005 | 2.063 ± 0.018 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.249 ± 0.018 | 0.304 ± 0.009 | 2.091 ± 0.023 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.409 ± 0.007 | 0.220 ± 0.054 | 1.645 ± 0.058 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.415 ± 0.012 | 0.252 ± 0.056 | 0.811 ± 0.017 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.423 ± 0.017 | 0.255 ± 0.010 | 1.674 ± 0.033 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.419 ± 0.011 | 0.251 ± 0.043 | 1.931 ± 0.842 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.421 ± 0.012 | 0.240 ± 0.006 | 1.926 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.435 ± 0.014 | 0.286 ± 0.011 | 2.913 ± 2.176 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.251 ± 0.008 | 0.409 ± 0.017 | 0.667 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.255 ± 0.010 | 0.409 ± 0.013 | 0.658 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.252 ± 0.008 | 0.408 ± 0.095 | 0.679 ± 0.031 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.271 ± 0.007 | 0.263 ± 0.058 | 1.415 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.280 ± 0.007 | 0.271 ± 0.055 | 1.409 ± 0.012 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.273 ± 0.006 | 0.266 ± 0.007 | 1.426 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.133 ± 0.006 | 0.249 ± 0.010 | 0.574 ± 0.022 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.141 ± 0.011 | 0.251 ± 0.006 | 0.599 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.153 ± 0.005 | 0.309 ± 0.013 | 0.578 ± 0.008 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.138 ± 0.006 | 0.206 ± 0.016 | 2.497 ± 1.749 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.141 ± 0.006 | 0.204 ± 0.008 | 1.749 ± 1.429 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.158 ± 0.013 | 0.257 ± 0.009 | 2.611 ± 1.616 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

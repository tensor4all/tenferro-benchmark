# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`
- Benchmark commit: `3f31f38e0491084b9414035754a3809ef0f5ae94`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_231715/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Timestamp: `20261027_231715`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261027_231715`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261027_231715/cpu_ops_t1_20261027_231715.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_231715/linalg_jvp_vjp_t1_20261027_231715.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 187.008 ± 0.705 | 258.292 ± 3.675 | 191.724 ± 0.427 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 4.286 ± 0.009 | 5.343 ± 0.010 | 4.439 ± 0.007 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 28.003 ± 0.035 | 36.983 ± 0.126 | 28.897 ± 0.042 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 188.994 ± 0.537 | 177.132 ± 0.839 | 190.045 ± 0.245 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.295 ± 0.003 | 4.327 ± 0.013 | 4.444 ± 0.010 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 27.836 ± 0.037 | 28.174 ± 0.086 | 28.650 ± 0.097 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 143.633 ± 0.723 | 311.117 ± 0.684 | 144.380 ± 1.637 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 2.654 ± 0.005 | 5.308 ± 0.081 | 2.528 ± 0.004 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 19.780 ± 0.127 | 46.582 ± 0.233 | 18.784 ± 0.037 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 143.876 ± 0.500 | 165.914 ± 1.192 | 144.087 ± 0.727 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.672 ± 0.002 | 2.819 ± 0.010 | 2.448 ± 0.003 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 19.376 ± 0.323 | 23.829 ± 0.127 | 18.262 ± 0.043 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 186.426 ± 0.292 | 220.816 ± 2.519 | 199.602 ± 1.597 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 3.672 ± 0.007 | 3.996 ± 0.023 | 3.515 ± 0.004 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 28.479 ± 0.134 | 33.753 ± 0.129 | 26.118 ± 0.064 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 184.472 ± 0.363 | 209.772 ± 2.942 | 197.539 ± 0.428 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 3.610 ± 0.008 | 4.008 ± 0.010 | 3.486 ± 0.007 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 26.629 ± 0.044 | 33.120 ± 0.478 | 26.300 ± 0.050 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 14.832 ± 0.062 | 22.067 ± 1.072 | 17.103 ± 0.106 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.349 ± 0.004 | 0.533 ± 0.007 | 0.361 ± 0.001 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 2.088 ± 0.007 | 3.175 ± 0.014 | 2.508 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 14.401 ± 0.065 | 36.808 ± 0.424 | 15.585 ± 0.052 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.350 ± 0.001 | 0.781 ± 0.009 | 0.367 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 2.077 ± 0.017 | 5.240 ± 0.008 | 2.512 ± 0.007 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 337.576 ± 6.623 | 466.134 ± 1.813 | 351.224 ± 0.688 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 7.617 ± 0.013 | 9.786 ± 0.014 | 7.833 ± 0.016 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 52.643 ± 0.257 | 73.323 ± 0.836 | 55.257 ± 0.106 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 337.412 ± 0.788 | 345.597 ± 2.802 | 358.294 ± 0.440 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.606 ± 0.034 | 8.183 ± 0.015 | 7.823 ± 0.011 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 52.158 ± 0.384 | 58.354 ± 0.212 | 55.221 ± 0.160 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.020 ± 0.000 | 0.060 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.010 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.048 ± 0.000 | 0.094 ± 0.001 | 0.054 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.048 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.013 ± 0.000 | 0.052 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.020 ± 0.000 | 0.052 ± 0.000 | 0.020 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.010 ± 0.000 | 0.041 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.048 ± 0.000 | 0.082 ± 0.000 | 0.055 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.041 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.013 ± 0.000 | 0.044 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.015 ± 0.000 | 0.083 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.012 ± 0.000 | 0.076 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.026 ± 0.000 | 0.104 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.012 ± 0.000 | 0.076 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.013 ± 0.000 | 0.077 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.015 ± 0.000 | 0.066 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.012 ± 0.000 | 0.061 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.027 ± 0.000 | 0.081 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.012 ± 0.000 | 0.061 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.012 ± 0.000 | 0.062 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.021 ± 0.000 | 0.060 ± 0.000 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.018 ± 0.000 | 0.054 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.035 ± 0.000 | 0.076 ± 0.001 | 0.024 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.018 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.018 ± 0.000 | 0.055 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.022 ± 0.000 | 0.063 ± 0.000 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.017 ± 0.000 | 0.056 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.036 ± 0.000 | 0.077 ± 0.000 | 0.024 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.018 ± 0.000 | 0.056 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.018 ± 0.000 | 0.057 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.009 ± 0.000 | 0.129 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.013 ± 0.000 | 0.135 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.009 ± 0.000 | 0.057 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.008 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.012 ± 0.000 | 0.064 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.008 ± 0.000 | 0.053 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.008 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.028 ± 0.000 | 0.100 ± 0.001 | 0.030 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.007 ± 0.000 | 0.071 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.077 ± 0.000 | 0.166 ± 0.002 | 0.080 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.009 ± 0.000 | 0.072 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.080 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.027 ± 0.000 | 0.069 ± 0.001 | 0.030 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.007 ± 0.000 | 0.044 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.077 ± 0.000 | 0.128 ± 0.001 | 0.080 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.008 ± 0.000 | 0.046 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.052 ± 0.000 | 0.013 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_235042/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Timestamp: `20261027_235042`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261027_235042`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261027_235042/cpu_ops_t4_20261027_235042.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_235042/linalg_jvp_vjp_t4_20261027_235042.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 65.816 ± 3.257 | 89.330 ± 1.765 | 105.188 ± 6.386 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.288 ± 0.060 | 2.371 ± 0.015 | 4.329 ± 0.250 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 31.235 ± 0.298 | 14.159 ± 0.046 | 21.770 ± 2.478 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 68.289 ± 0.484 | 60.836 ± 0.348 | 107.397 ± 5.921 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.192 ± 0.017 | 2.055 ± 0.012 | 4.185 ± 0.237 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 11.145 ± 3.104 | 11.005 ± 0.052 | 21.794 ± 3.088 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 49.739 ± 0.263 | 115.178 ± 0.901 | 84.976 ± 6.681 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.988 ± 0.017 | 2.229 ± 0.032 | 2.356 ± 0.490 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 9.787 ± 0.070 | 18.516 ± 0.194 | 14.852 ± 2.703 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 51.512 ± 0.269 | 55.290 ± 0.343 | 80.030 ± 6.903 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.527 ± 0.015 | 1.082 ± 0.020 | 1.845 ± 0.322 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 7.225 ± 0.090 | 8.289 ± 0.085 | 12.981 ± 2.093 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 70.481 ± 39.705 | 73.734 ± 0.357 | 122.938 ± 14.901 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.221 ± 0.043 | 1.668 ± 0.008 | 3.324 ± 0.759 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 11.850 ± 0.044 | 12.310 ± 0.039 | 18.769 ± 3.611 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 69.388 ± 1.124 | 72.653 ± 0.361 | 112.952 ± 6.229 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 8.337 ± 1.216 | 1.795 ± 0.037 | 3.417 ± 0.476 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 10.644 ± 0.059 | 13.951 ± 0.067 | 19.929 ± 4.984 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 8.963 ± 0.091 | 5.662 ± 0.070 | 9.059 ± 0.308 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.453 ± 0.006 | 0.295 ± 0.068 | 0.298 ± 0.010 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 0.928 ± 0.004 | 1.121 ± 2.991 | 1.635 ± 0.013 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 9.737 ± 0.186 | 10.406 ± 0.089 | 8.167 ± 0.046 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.447 ± 0.003 | 0.344 ± 0.009 | 0.315 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 0.901 ± 0.002 | 1.677 ± 2.995 | 1.663 ± 0.009 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 137.813 ± 0.616 | 191.763 ± 1.988 | 214.261 ± 5.431 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.343 ± 0.010 | 6.379 ± 0.020 | 7.250 ± 0.131 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 26.240 ± 0.117 | 34.944 ± 0.152 | 43.837 ± 2.596 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 138.654 ± 0.909 | 151.604 ± 1.104 | 215.124 ± 12.100 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 6.016 ± 0.236 | 5.880 ± 0.013 | 7.402 ± 0.099 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 26.587 ± 0.321 | 31.007 ± 0.125 | 43.314 ± 1.597 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.021 ± 0.000 | 0.061 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.012 ± 0.000 | 0.049 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.054 ± 0.000 | 0.098 ± 0.001 | 0.052 ± 0.005 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.012 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.052 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.021 ± 0.000 | 0.052 ± 0.000 | 0.017 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.011 ± 0.000 | 0.041 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.054 ± 0.000 | 0.086 ± 0.000 | 0.051 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.012 ± 0.000 | 0.042 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.045 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.017 ± 0.000 | 0.087 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.000 | 0.082 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.030 ± 0.000 | 0.108 ± 0.001 | 0.021 ± 0.018 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.014 ± 0.000 | 0.081 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.015 ± 0.000 | 0.083 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.016 ± 0.000 | 0.069 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.000 | 0.065 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.037 ± 0.000 | 0.083 ± 0.001 | 0.025 ± 0.015 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.013 ± 0.000 | 0.065 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.066 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.024 ± 0.000 | 0.064 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.021 ± 0.000 | 0.058 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.040 ± 0.000 | 0.078 ± 0.001 | 0.046 ± 0.125 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.021 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.021 ± 0.000 | 0.059 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.024 ± 0.000 | 0.067 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.020 ± 0.000 | 0.059 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.039 ± 0.000 | 0.080 ± 0.001 | 0.024 ± 0.006 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.020 ± 0.000 | 0.059 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.021 ± 0.000 | 0.061 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.010 ± 0.000 | 0.130 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.009 ± 0.000 | 0.129 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.014 ± 0.000 | 0.136 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.009 ± 0.000 | 0.128 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.009 ± 0.000 | 0.130 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.010 ± 0.000 | 0.057 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.009 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.014 ± 0.000 | 0.064 ± 0.000 | 0.008 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.009 ± 0.000 | 0.055 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.009 ± 0.000 | 0.057 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.029 ± 0.000 | 0.101 ± 0.001 | 0.026 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.008 ± 0.000 | 0.075 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.085 ± 0.000 | 0.172 ± 0.002 | 0.077 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.010 ± 0.000 | 0.074 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.015 ± 0.000 | 0.079 ± 0.009 | 0.009 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.028 ± 0.000 | 0.070 ± 0.001 | 0.026 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.008 ± 0.000 | 0.046 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.084 ± 0.000 | 0.133 ± 0.001 | 0.077 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.009 ± 0.000 | 0.047 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.014 ± 0.000 | 0.054 ± 0.001 | 0.009 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261028_001612/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Timestamp: `20261028_001612`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261028_001612`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261028_001612/cpu_ops_t8_20261028_001612.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261028_001612/linalg_jvp_vjp_t8_20261028_001612.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | 145.846 ± 71.297 | 88.704 ± 0.388 | 131.017 ± 14.524 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | 23.445 ± 0.233 | 2.969 ± 0.032 | 7.389 ± 0.450 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | 39.714 ± 16.619 | 13.546 ± 0.278 | 33.056 ± 3.104 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | 70.888 ± 62.232 | 61.875 ± 0.340 | 127.451 ± 7.475 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | 23.683 ± 0.036 | 2.624 ± 0.024 | 7.439 ± 0.551 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | 56.452 ± 0.163 | 10.593 ± 0.061 | 30.960 ± 2.552 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | 72.732 ± 7.440 | 105.859 ± 1.208 | 78.793 ± 23.053 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | 3.673 ± 0.256 | 2.608 ± 0.035 | 3.972 ± 0.802 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | 13.972 ± 0.074 | 17.580 ± 0.324 | 20.051 ± 7.398 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | 80.207 ± 2.773 | 52.381 ± 0.437 | 87.642 ± 10.241 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | 5.358 ± 0.448 | 1.352 ± 0.030 | 3.727 ± 0.760 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | 13.830 ± 0.152 | 8.620 ± 0.095 | 16.255 ± 4.643 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | 72.724 ± 0.951 | 75.625 ± 0.350 | 130.987 ± 18.173 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | 11.764 ± 0.257 | 2.129 ± 0.033 | 6.683 ± 1.141 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | 26.142 ± 0.685 | 11.916 ± 0.041 | 27.349 ± 5.019 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | 162.127 ± 60.281 | 72.048 ± 0.613 | 126.977 ± 16.489 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | 12.632 ± 0.378 | 2.294 ± 0.018 | 6.924 ± 1.097 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | 21.594 ± 0.283 | 13.993 ± 0.229 | 28.212 ± 4.203 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | 32.936 ± 33.342 | 6.222 ± 0.079 | 9.895 ± 0.620 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | 1.771 ± 0.004 | 0.921 ± 1.365 | 0.348 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | 2.128 ± 0.027 | 1.411 ± 0.056 | 2.285 ± 0.871 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | 15.299 ± 26.607 | 10.180 ± 0.209 | 8.984 ± 0.301 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | 1.767 ± 0.004 | 0.553 ± 0.024 | 0.354 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | 2.643 ± 0.009 | 1.984 ± 0.012 | 2.241 ± 0.063 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | 153.806 ± 1.073 | 212.593 ± 3.732 | 286.415 ± 13.652 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | 14.975 ± 0.191 | 9.340 ± 0.056 | 12.494 ± 0.835 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | 61.152 ± 20.178 | 42.266 ± 0.200 | 64.028 ± 7.731 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | 154.665 ± 1.333 | 176.369 ± 4.091 | 293.920 ± 13.207 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | 15.362 ± 0.549 | 8.582 ± 0.029 | 11.971 ± 0.956 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | 61.142 ± 0.590 | 40.990 ± 1.113 | 67.046 ± 3.091 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | 0.034 ± 0.000 | 0.095 ± 0.001 | 0.026 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | 0.018 ± 0.000 | 0.075 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | 0.085 ± 0.000 | 0.155 ± 0.003 | 0.084 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | 0.020 ± 0.000 | 0.076 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | 0.023 ± 0.000 | 0.080 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | 0.034 ± 0.000 | 0.081 ± 0.001 | 0.026 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | 0.018 ± 0.000 | 0.063 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | 0.085 ± 0.000 | 0.135 ± 0.002 | 0.080 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | 0.019 ± 0.000 | 0.064 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | 0.022 ± 0.000 | 0.067 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | 0.027 ± 0.000 | 0.138 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | 0.022 ± 0.000 | 0.127 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | 0.051 ± 0.000 | 0.209 ± 0.003 | 0.061 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | 0.023 ± 0.000 | 0.125 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | 0.023 ± 0.000 | 0.127 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | 0.026 ± 0.000 | 0.108 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | 0.021 ± 0.000 | 0.099 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | 0.061 ± 0.000 | 0.135 ± 0.002 | 0.034 ± 0.126 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | 0.021 ± 0.000 | 0.099 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | 0.022 ± 0.000 | 0.101 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | 0.038 ± 0.000 | 0.099 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | 0.032 ± 0.000 | 0.088 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | 0.062 ± 0.000 | 0.130 ± 0.001 | 0.249 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | 0.032 ± 0.000 | 0.087 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | 0.034 ± 0.000 | 0.090 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | 0.038 ± 0.000 | 0.102 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | 0.031 ± 0.000 | 0.090 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | 0.061 ± 0.000 | 0.142 ± 0.001 | 0.046 ± 0.066 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | 0.032 ± 0.000 | 0.090 ± 0.000 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | 0.033 ± 0.000 | 0.093 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | 0.016 ± 0.000 | 0.206 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | 0.014 ± 0.000 | 0.200 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | 0.022 ± 0.000 | 0.215 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | 0.015 ± 0.000 | 0.200 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.202 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | 0.016 ± 0.000 | 0.089 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | 0.014 ± 0.000 | 0.084 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | 0.022 ± 0.000 | 0.102 ± 0.001 | 0.013 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | 0.014 ± 0.000 | 0.084 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | 0.015 ± 0.000 | 0.086 ± 0.000 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | 0.045 ± 0.000 | 0.159 ± 0.005 | 0.041 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | 0.013 ± 0.000 | 0.113 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | 0.134 ± 0.000 | 0.271 ± 0.004 | 0.123 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | 0.015 ± 0.000 | 0.115 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | 0.024 ± 0.000 | 0.127 ± 0.001 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | 0.045 ± 0.000 | 0.108 ± 0.001 | 0.042 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | 0.012 ± 0.000 | 0.069 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | 0.133 ± 0.000 | 0.208 ± 0.003 | 0.122 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | 0.014 ± 0.000 | 0.071 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | 0.023 ± 0.000 | 0.081 ± 0.001 | 0.014 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

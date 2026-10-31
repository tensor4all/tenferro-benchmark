# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_095941/run.yaml`
- Timestamp: `2026-10-31T00:59:41.343309+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261031_095941`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_095941/cpu_ops_t4_20261031_095941.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 218.750 ± 28.562 | 202.799 ± 1.748 | 254.797 ± 7.844 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 21.337 ± 3.995 | 6.068 ± 0.058 | 11.115 ± 0.748 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 50.997 ± 0.009 | 34.421 ± 0.221 | 50.758 ± 3.351 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 221.998 ± 31.840 | 140.038 ± 2.217 | 252.164 ± 6.496 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 36.000 ± 5.008 | 5.253 ± 0.022 | 11.018 ± 0.538 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 50.999 ± 0.009 | 25.018 ± 0.093 | 51.596 ± 2.669 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 166.996 ± 2.020 | 241.185 ± 5.199 | 178.450 ± 18.264 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 57.010 ± 31.001 | 5.603 ± 0.091 | 4.836 ± 1.239 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 56.069 ± 32.001 | 43.361 ± 0.173 | 29.992 ± 3.189 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 162.363 ± 6.890 | 115.886 ± 0.853 | 173.572 ± 17.242 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 36.998 ± 27.337 | 2.687 ± 0.036 | 4.492 ± 1.569 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 77.025 ± 2.032 | 18.473 ± 0.486 | 30.098 ± 5.089 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 198.282 ± 43.369 | 163.496 ± 4.116 | 261.279 ± 14.294 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 58.991 ± 53.006 | 4.245 ± 0.031 | 8.255 ± 1.808 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 123.001 ± 20.005 | 28.891 ± 0.079 | 46.081 ± 4.783 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 210.987 ± 48.024 | 158.121 ± 0.617 | 247.014 ± 10.014 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 56.986 ± 39.981 | 4.519 ± 0.038 | 8.710 ± 1.279 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 134.997 ± 7.973 | 31.387 ± 0.340 | 46.304 ± 5.287 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 24.282 ± 0.051 | 12.857 ± 0.072 | 20.179 ± 1.344 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 17.000 ± 0.002 | 0.771 ± 0.041 | 0.771 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 12.987 ± 0.021 | 2.872 ± 0.151 | 4.214 ± 0.027 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 31.999 ± 0.003 | 22.395 ± 0.209 | 19.413 ± 0.285 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 17.000 ± 2.996 | 0.869 ± 0.023 | 0.782 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 13.070 ± 0.021 | 4.326 ± 0.054 | 4.289 ± 0.060 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 412.961 ± 10.422 | 436.993 ± 6.326 | 523.162 ± 12.414 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 25.100 ± 0.048 | 16.468 ± 0.072 | 18.911 ± 0.697 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 95.484 ± 1.459 | 85.515 ± 0.320 | 109.892 ± 2.768 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 431.022 ± 2.908 | 351.521 ± 1.977 | 524.287 ± 5.736 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 39.002 ± 3.001 | 15.331 ± 0.053 | 18.535 ± 0.775 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 105.588 ± 2.698 | 82.218 ± 4.902 | 107.110 ± 4.054 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | 0.068 ± 0.000 | 0.160 ± 0.005 | 0.042 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.033 ± 0.000 | 0.123 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | 0.158 ± 0.000 | 0.259 ± 0.005 | 0.131 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.035 ± 0.000 | 0.126 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.042 ± 0.000 | 0.136 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | 0.067 ± 0.000 | 0.135 ± 0.002 | 0.043 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.031 ± 0.000 | 0.106 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | 0.157 ± 0.001 | 0.222 ± 0.004 | 0.133 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.034 ± 0.000 | 0.108 ± 0.000 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.040 ± 0.000 | 0.115 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | 0.046 ± 0.000 | 0.228 ± 0.004 | 0.021 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.040 ± 0.000 | 0.211 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | 0.063 ± 0.000 | 0.281 ± 0.003 | 0.059 ± 0.019 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.040 ± 0.000 | 0.208 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.042 ± 0.000 | 0.213 ± 0.003 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | 0.044 ± 0.000 | 0.182 ± 0.012 | 0.019 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.037 ± 0.000 | 0.163 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | 0.065 ± 0.000 | 0.214 ± 0.005 | 0.054 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.037 ± 0.000 | 0.163 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.039 ± 0.000 | 0.169 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | 0.068 ± 0.000 | 0.165 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.057 ± 0.000 | 0.143 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | 0.099 ± 0.000 | 0.205 ± 0.005 | 0.076 ± 0.028 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.058 ± 0.000 | 0.142 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.060 ± 0.000 | 0.151 ± 0.005 | 0.020 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | 0.067 ± 0.000 | 0.167 ± 0.007 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.056 ± 0.000 | 0.151 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | 0.100 ± 0.000 | 0.209 ± 0.006 | 0.068 ± 0.016 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.057 ± 0.000 | 0.151 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.060 ± 0.000 | 0.154 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.337 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.026 ± 0.000 | 0.327 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | 0.036 ± 0.000 | 0.354 ± 0.008 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.027 ± 0.000 | 0.326 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.028 ± 0.000 | 0.329 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | 0.029 ± 0.000 | 0.148 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.026 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | 0.036 ± 0.000 | 0.167 ± 0.005 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.026 ± 0.000 | 0.141 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | 0.081 ± 0.000 | 0.264 ± 0.004 | 0.069 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.023 ± 0.000 | 0.188 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | 0.216 ± 0.000 | 0.452 ± 0.004 | 0.205 ± 0.004 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.026 ± 0.000 | 0.194 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.037 ± 0.000 | 0.211 ± 0.005 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | 0.079 ± 0.000 | 0.178 ± 0.005 | 0.069 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.021 ± 0.000 | 0.115 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | 0.214 ± 0.000 | 0.351 ± 0.011 | 0.205 ± 0.004 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.024 ± 0.000 | 0.121 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.035 ± 0.000 | 0.136 ± 0.001 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

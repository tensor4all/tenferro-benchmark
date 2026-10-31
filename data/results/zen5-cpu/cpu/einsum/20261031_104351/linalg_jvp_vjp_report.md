# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_104351/run.yaml`
- Timestamp: `2026-10-31T01:43:51.562747+00:00`

Latest run: `./scripts/run_all.sh 8`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261031_104351`.

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_104351/cpu_ops_t8_20261031_104351.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | 189.998 ± 35.440 | 133.390 ± 0.579 | 206.114 ± 17.387 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | 53.968 ± 5.998 | 4.842 ± 0.048 | 11.816 ± 1.752 |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | 89.992 ± 15.211 | 21.225 ± 0.166 | 46.909 ± 5.578 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | 188.997 ± 6.165 | 96.324 ± 0.423 | 200.827 ± 12.909 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | 58.999 ± 9.000 | 4.358 ± 0.038 | 11.510 ± 0.875 |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | 92.996 ± 24.982 | 16.950 ± 0.126 | 47.014 ± 3.511 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | 128.989 ± 30.015 | 146.653 ± 1.455 | 121.086 ± 7.952 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | 74.000 ± 12.142 | 4.113 ± 0.042 | 6.967 ± 3.623 |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | 88.999 ± 27.714 | 27.335 ± 0.249 | 23.690 ± 6.726 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | 127.206 ± 20.827 | 72.352 ± 0.181 | 120.879 ± 18.902 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | 53.034 ± 24.660 | 2.120 ± 0.059 | 6.375 ± 3.009 |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | 85.980 ± 18.445 | 11.595 ± 0.084 | 22.187 ± 8.472 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | 211.007 ± 25.983 | 110.665 ± 0.384 | 195.889 ± 20.533 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | 95.998 ± 19.002 | 3.438 ± 0.030 | 9.597 ± 2.868 |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | 138.003 ± 10.990 | 20.611 ± 0.147 | 39.287 ± 6.327 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | 227.997 ± 37.002 | 105.321 ± 0.507 | 186.900 ± 14.890 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | 68.998 ± 18.999 | 3.672 ± 0.030 | 8.770 ± 3.378 |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | 118.985 ± 18.993 | 22.392 ± 0.182 | 43.959 ± 12.119 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | 19.183 ± 0.063 | 9.849 ± 0.056 | 15.058 ± 0.388 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | 19.999 ± 0.011 | 0.791 ± 0.013 | 0.576 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | 12.404 ± 0.021 | 2.027 ± 0.045 | 3.678 ± 0.048 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | 26.999 ± 3.000 | 14.977 ± 0.102 | 14.489 ± 1.408 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | 20.000 ± 0.006 | 0.896 ± 0.029 | 0.589 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | 12.387 ± 0.261 | 3.029 ± 0.019 | 3.665 ± 0.183 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | 341.597 ± 8.461 | 328.967 ± 1.306 | 470.148 ± 23.539 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | 36.632 ± 6.658 | 15.159 ± 0.050 | 19.293 ± 1.919 |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | 85.046 ± 0.183 | 68.653 ± 0.181 | 102.943 ± 2.980 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | 351.400 ± 6.011 | 278.482 ± 0.691 | 465.000 ± 15.933 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | 43.139 ± 9.722 | 14.187 ± 0.028 | 19.654 ± 0.961 |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | 98.002 ± 1.015 | 64.073 ± 0.115 | 104.326 ± 6.176 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | 0.069 ± 0.000 | 0.161 ± 0.005 | 0.042 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | 0.033 ± 0.000 | 0.126 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | 0.160 ± 0.000 | 0.263 ± 0.005 | 0.132 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | 0.036 ± 0.000 | 0.127 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | 0.043 ± 0.000 | 0.135 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | 0.068 ± 0.000 | 0.137 ± 0.001 | 0.043 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | 0.032 ± 0.000 | 0.108 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | 0.160 ± 0.000 | 0.229 ± 0.005 | 0.133 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | 0.034 ± 0.000 | 0.109 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | 0.041 ± 0.000 | 0.116 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | 0.047 ± 0.000 | 0.233 ± 0.004 | 0.021 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | 0.041 ± 0.000 | 0.214 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | 0.065 ± 0.000 | 0.350 ± 0.005 | 0.122 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | 0.041 ± 0.000 | 0.213 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | 0.043 ± 0.000 | 0.219 ± 0.005 | 0.022 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | 0.045 ± 0.000 | 0.184 ± 0.005 | 0.019 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | 0.037 ± 0.000 | 0.167 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | 0.066 ± 0.000 | 0.232 ± 0.007 | 0.071 ± 0.202 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | 0.038 ± 0.000 | 0.167 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | 0.040 ± 0.000 | 0.173 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | 0.069 ± 0.000 | 0.169 ± 0.007 | 0.019 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | 0.058 ± 0.000 | 0.146 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | 0.100 ± 0.000 | 0.218 ± 0.006 | 0.248 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | 0.060 ± 0.000 | 0.147 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | 0.062 ± 0.000 | 0.153 ± 0.004 | 0.020 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | 0.069 ± 0.000 | 0.176 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | 0.057 ± 0.000 | 0.156 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | 0.102 ± 0.001 | 0.241 ± 0.006 | 0.069 ± 0.063 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | 0.059 ± 0.000 | 0.156 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | 0.062 ± 0.000 | 0.159 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.342 ± 0.003 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | 0.027 ± 0.000 | 0.328 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | 0.037 ± 0.000 | 0.354 ± 0.005 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | 0.027 ± 0.000 | 0.328 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | 0.028 ± 0.000 | 0.333 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | 0.030 ± 0.000 | 0.151 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | 0.037 ± 0.000 | 0.169 ± 0.005 | 0.020 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | 0.028 ± 0.000 | 0.145 ± 0.001 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | 0.082 ± 0.000 | 0.266 ± 0.005 | 0.069 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | 0.023 ± 0.000 | 0.185 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | 0.217 ± 0.000 | 0.456 ± 0.005 | 0.203 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | 0.027 ± 0.000 | 0.194 ± 0.007 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | 0.038 ± 0.000 | 0.213 ± 0.006 | 0.022 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | 0.080 ± 0.000 | 0.183 ± 0.006 | 0.069 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | 0.021 ± 0.000 | 0.117 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | 0.215 ± 0.001 | 0.356 ± 0.005 | 0.203 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | 0.025 ± 0.000 | 0.122 ± 0.001 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | 0.036 ± 0.000 | 0.139 ± 0.001 | 0.022 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

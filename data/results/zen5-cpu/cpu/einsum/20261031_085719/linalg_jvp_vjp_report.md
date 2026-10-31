# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_085719/run.yaml`
- Timestamp: `2026-10-30T23:57:19.436239+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/zen5-cpu/cpu/einsum/20261031_085719`.

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_085719/cpu_ops_t1_20261031_085719.csv`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | 389.396 ± 0.794 | 624.841 ± 2.643 | 480.201 ± 1.613 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 10.475 ± 0.015 | 14.023 ± 0.056 | 11.755 ± 0.045 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 63.148 ± 0.078 | 93.677 ± 0.256 | 75.632 ± 0.517 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | 391.148 ± 1.035 | 450.112 ± 0.793 | 480.646 ± 1.241 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 10.532 ± 0.021 | 11.331 ± 0.020 | 11.738 ± 0.016 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 63.194 ± 0.188 | 72.612 ± 0.116 | 74.227 ± 0.218 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | 375.255 ± 0.358 | 752.922 ± 1.399 | 353.788 ± 1.223 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 7.149 ± 0.006 | 13.835 ± 0.052 | 6.798 ± 0.022 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 50.740 ± 0.083 | 107.067 ± 0.150 | 47.387 ± 0.061 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | 378.214 ± 0.327 | 399.549 ± 0.808 | 349.546 ± 1.070 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 7.315 ± 0.026 | 7.426 ± 0.044 | 6.448 ± 0.059 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 51.424 ± 0.064 | 56.029 ± 0.664 | 46.578 ± 0.083 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | 465.428 ± 0.871 | 518.404 ± 1.995 | 497.811 ± 1.566 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 9.397 ± 0.032 | 10.501 ± 0.036 | 9.276 ± 0.036 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 66.534 ± 0.083 | 79.651 ± 0.300 | 67.254 ± 0.107 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | 466.968 ± 0.863 | 505.869 ± 0.819 | 497.327 ± 4.005 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 9.485 ± 0.020 | 10.551 ± 0.036 | 9.207 ± 0.036 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 66.540 ± 0.286 | 83.520 ± 0.091 | 68.021 ± 0.198 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | 35.734 ± 0.111 | 48.410 ± 0.175 | 41.192 ± 0.070 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.881 ± 0.003 | 1.388 ± 0.014 | 0.950 ± 0.002 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 5.269 ± 0.019 | 7.551 ± 0.030 | 6.569 ± 0.026 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | 35.699 ± 0.083 | 84.417 ± 0.224 | 40.316 ± 0.057 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.886 ± 0.007 | 2.039 ± 0.015 | 0.967 ± 0.002 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.330 ± 0.084 | 12.948 ± 0.076 | 6.610 ± 0.025 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | 862.054 ± 9.461 | 1137.682 ± 3.463 | 895.931 ± 1.944 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 20.391 ± 0.039 | 25.723 ± 0.068 | 20.662 ± 0.035 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 140.448 ± 0.201 | 189.718 ± 0.452 | 142.662 ± 0.284 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | 862.034 ± 0.573 | 869.346 ± 5.966 | 913.789 ± 3.374 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 20.370 ± 0.034 | 21.500 ± 0.056 | 20.631 ± 0.048 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 140.538 ± 0.319 | 161.918 ± 1.178 | 143.724 ± 0.228 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | 0.061 ± 0.000 | 0.161 ± 0.005 | 0.053 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.026 ± 0.000 | 0.123 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | 0.151 ± 0.000 | 0.253 ± 0.004 | 0.142 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.028 ± 0.000 | 0.125 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.035 ± 0.000 | 0.134 ± 0.002 | 0.025 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | 0.061 ± 0.000 | 0.135 ± 0.001 | 0.053 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.025 ± 0.000 | 0.106 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | 0.151 ± 0.000 | 0.221 ± 0.005 | 0.142 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.027 ± 0.000 | 0.107 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.034 ± 0.000 | 0.113 ± 0.001 | 0.026 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | 0.037 ± 0.000 | 0.223 ± 0.006 | 0.026 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.031 ± 0.000 | 0.202 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | 0.059 ± 0.000 | 0.276 ± 0.006 | 0.051 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.031 ± 0.000 | 0.201 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.033 ± 0.000 | 0.204 ± 0.004 | 0.021 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | 0.037 ± 0.000 | 0.174 ± 0.005 | 0.027 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.029 ± 0.000 | 0.159 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | 0.060 ± 0.000 | 0.214 ± 0.005 | 0.052 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.029 ± 0.000 | 0.160 ± 0.005 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.031 ± 0.000 | 0.161 ± 0.006 | 0.021 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | 0.056 ± 0.000 | 0.161 ± 0.006 | 0.030 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.045 ± 0.000 | 0.139 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | 0.092 ± 0.000 | 0.203 ± 0.005 | 0.062 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.046 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.048 ± 0.000 | 0.142 ± 0.002 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | 0.056 ± 0.000 | 0.164 ± 0.004 | 0.030 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.044 ± 0.000 | 0.147 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | 0.093 ± 0.000 | 0.206 ± 0.004 | 0.062 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.045 ± 0.000 | 0.148 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.048 ± 0.000 | 0.150 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | 0.024 ± 0.000 | 0.340 ± 0.001 | 0.018 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.021 ± 0.000 | 0.327 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | 0.031 ± 0.008 | 0.358 ± 0.004 | 0.029 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.021 ± 0.000 | 0.327 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.022 ± 0.000 | 0.329 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | 0.024 ± 0.000 | 0.149 ± 0.001 | 0.020 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.020 ± 0.000 | 0.142 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | 0.031 ± 0.000 | 0.170 ± 0.004 | 0.033 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.021 ± 0.000 | 0.141 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.021 ± 0.000 | 0.144 ± 0.001 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | 0.078 ± 0.000 | 0.266 ± 0.003 | 0.078 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.019 ± 0.000 | 0.185 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | 0.211 ± 0.001 | 0.442 ± 0.007 | 0.209 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.022 ± 0.000 | 0.194 ± 0.007 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.034 ± 0.000 | 0.212 ± 0.006 | 0.033 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | 0.076 ± 0.000 | 0.182 ± 0.005 | 0.078 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.018 ± 0.000 | 0.118 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | 0.211 ± 0.001 | 0.344 ± 0.008 | 0.211 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.021 ± 0.000 | 0.121 ± 0.002 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.032 ± 0.000 | 0.136 ± 0.001 | 0.034 ± 0.000 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

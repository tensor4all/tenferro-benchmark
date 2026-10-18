# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`
- Benchmark commit: `e4b601dcd685fe1bafd45e6da97bc134f45c5049`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261018_180413/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261018_180413`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261018_180413`.

- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261018_180413/cpu_ops_t1_20261018_180413.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261018_180413/linalg_jvp_vjp_t1_20261018_180413.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 8.649 ± 0.073 | 10.334 ± 0.179 | 9.249 ± 0.043 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 57.293 ± 0.186 | 67.834 ± 0.315 | 61.819 ± 0.370 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.693 ± 0.048 | 8.462 ± 0.072 | 9.146 ± 0.081 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 57.096 ± 0.849 | 52.891 ± 0.718 | 62.043 ± 0.478 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 5.132 ± 0.061 | 12.681 ± 0.265 | 4.549 ± 0.071 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 35.872 ± 0.347 | 80.862 ± 0.496 | 34.659 ± 0.314 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.366 ± 0.072 | 5.508 ± 0.098 | 4.715 ± 0.096 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 35.900 ± 0.220 | 42.937 ± 1.000 | 34.582 ± 0.740 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 7.379 ± 0.119 | 7.487 ± 0.201 | 7.395 ± 0.157 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 54.094 ± 0.255 | 54.591 ± 0.419 | 50.137 ± 2.244 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.146 ± 0.023 | 7.346 ± 0.034 | 7.646 ± 0.193 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 51.383 ± 0.193 | 53.802 ± 0.288 | 55.703 ± 0.930 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.771 ± 0.005 | 1.284 ± 0.151 | 0.702 ± 0.009 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 4.334 ± 0.084 | 5.915 ± 0.304 | 4.242 ± 0.295 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.756 ± 0.003 | 1.737 ± 0.026 | 0.701 ± 0.013 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.311 ± 0.057 | 9.686 ± 0.109 | 4.149 ± 0.052 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 15.225 ± 0.274 | 21.157 ± 0.243 | 15.915 ± 0.232 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 104.662 ± 0.903 | 136.886 ± 4.362 | 120.093 ± 0.884 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 15.061 ± 0.169 | 16.408 ± 0.397 | 16.772 ± 0.287 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 104.941 ± 0.301 | 115.134 ± 1.311 | 121.918 ± 0.911 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.054 ± 0.001 | 0.197 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.060 ± 0.000 | 0.202 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.067 ± 0.002 | 0.209 ± 0.004 | 0.013 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.056 ± 0.000 | 0.196 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.062 ± 0.000 | 0.204 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.067 ± 0.000 | 0.210 ± 0.004 | 0.014 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.053 ± 0.002 | 0.273 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.052 ± 0.001 | 0.278 ± 0.006 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.054 ± 0.001 | 0.281 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.055 ± 0.001 | 0.263 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.055 ± 0.003 | 0.265 ± 0.088 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.056 ± 0.000 | 0.265 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.090 ± 0.001 | 0.222 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.092 ± 0.002 | 0.222 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.092 ± 0.001 | 0.223 ± 0.005 | 0.017 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.089 ± 0.001 | 0.250 ± 0.005 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.091 ± 0.002 | 0.250 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.092 ± 0.003 | 0.253 ± 0.005 | 0.014 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.033 ± 0.001 | 0.415 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.032 ± 0.001 | 0.420 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.034 ± 0.000 | 0.420 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.032 ± 0.000 | 0.251 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.032 ± 0.000 | 0.249 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.033 ± 0.000 | 0.250 ± 0.008 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.034 ± 0.000 | 0.265 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.040 ± 0.000 | 0.271 ± 0.010 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.051 ± 0.001 | 0.290 ± 0.005 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.001 | 0.222 ± 0.021 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.039 ± 0.000 | 0.225 ± 0.009 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.051 ± 0.000 | 0.236 ± 0.009 | 0.019 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261018_183622/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Timestamp: `20261018_183622`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261018_183622`.

- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261018_183622/cpu_ops_t4_20261018_183622.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261018_183622/linalg_jvp_vjp_t4_20261018_183622.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 6.796 ± 0.057 | 7.113 ± 0.205 | 10.217 ± 0.887 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 33.796 ± 0.212 | 41.223 ± 2.146 | 61.371 ± 0.811 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 6.671 ± 0.056 | 6.090 ± 0.156 | 10.999 ± 1.063 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 34.470 ± 0.855 | 31.395 ± 0.506 | 60.819 ± 0.864 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.460 ± 0.058 | 8.633 ± 0.566 | 6.035 ± 0.590 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 19.526 ± 0.262 | 48.378 ± 1.165 | 37.128 ± 6.769 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 3.881 ± 0.111 | 3.700 ± 0.086 | 5.906 ± 1.504 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 19.788 ± 0.427 | 24.277 ± 0.214 | 30.321 ± 2.236 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 5.396 ± 0.068 | 4.923 ± 0.093 | 10.184 ± 1.115 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 31.440 ± 0.351 | 33.530 ± 0.569 | 58.538 ± 10.529 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.369 ± 0.181 | 5.472 ± 0.060 | 9.756 ± 1.935 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 29.310 ± 0.184 | 34.522 ± 0.865 | 60.255 ± 3.714 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.730 ± 0.016 | 5.987 ± 2.955 | 0.908 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 2.570 ± 0.076 | 3.100 ± 0.079 | 3.859 ± 0.043 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.695 ± 0.009 | 1.253 ± 0.042 | 0.947 ± 0.037 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.588 ± 0.026 | 4.800 ± 0.060 | 3.720 ± 0.044 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 13.451 ± 0.291 | 16.644 ± 0.180 | 17.518 ± 2.047 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 78.418 ± 1.604 | 91.939 ± 0.966 | 108.005 ± 7.588 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 13.260 ± 0.159 | 14.832 ± 0.223 | 17.784 ± 1.028 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 77.252 ± 1.785 | 82.567 ± 1.569 | 108.785 ± 1.688 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.300 ± 0.057 | 0.200 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.347 ± 0.045 | 0.207 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.335 ± 0.038 | 0.211 ± 0.002 | 0.017 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.298 ± 0.042 | 0.196 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.279 ± 0.009 | 0.207 ± 0.009 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.307 ± 0.011 | 0.206 ± 0.006 | 0.019 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.311 ± 0.012 | 0.382 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.262 ± 0.024 | 0.380 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.273 ± 0.024 | 0.384 ± 0.004 | 0.022 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.271 ± 0.021 | 0.337 ± 0.011 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.263 ± 0.017 | 0.342 ± 0.085 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.273 ± 0.011 | 0.339 ± 0.010 | 0.021 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.482 ± 0.010 | 0.291 ± 0.011 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.475 ± 0.031 | 0.285 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.520 ± 0.035 | 0.292 ± 0.004 | 0.021 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.473 ± 0.091 | 0.320 ± 0.012 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.454 ± 0.068 | 0.318 ± 0.006 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.485 ± 0.030 | 0.325 ± 0.005 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.198 ± 0.016 | 0.417 ± 0.003 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.200 ± 0.016 | 0.430 ± 0.015 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.202 ± 0.008 | 0.418 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.165 ± 0.016 | 0.249 ± 0.003 | 0.014 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.195 ± 0.004 | 0.248 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.165 ± 0.006 | 0.252 ± 0.009 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.149 ± 0.004 | 0.267 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.165 ± 0.002 | 0.274 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.206 ± 0.007 | 0.289 ± 0.002 | 0.026 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.144 ± 0.008 | 0.218 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.164 ± 0.003 | 0.221 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.179 ± 0.010 | 0.238 ± 0.005 | 0.035 ± 0.004 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `9192ee1056de2203697691c63cc9ea04b0c5298e`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261104_205416/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261104_205416/run.yaml`
- Timestamp: `2026-11-04T20:54:16.992391+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261104_205416`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261104_205416/cpu_ops_t1_20261104_205416.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 17.100 ± 0.262 | 8.718 ± 0.100 | 10.965 ± 0.193 | 9.260 ± 0.064 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 105.186 ± 0.910 | 55.710 ± 0.732 | 67.171 ± 3.352 | 59.194 ± 0.507 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 10.217 ± 0.169 | 8.776 ± 0.051 | 9.013 ± 0.031 | 9.113 ± 0.167 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 62.786 ± 0.340 | 55.940 ± 0.635 | 52.584 ± 0.191 | 57.862 ± 0.339 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 6.486 ± 0.384 | 5.068 ± 0.109 | 13.380 ± 0.230 | 4.427 ± 0.093 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 41.466 ± 0.664 | 36.417 ± 0.951 | 81.324 ± 1.081 | 31.655 ± 1.402 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 5.925 ± 0.118 | 5.382 ± 0.120 | 5.731 ± 0.217 | 4.433 ± 0.045 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 39.928 ± 0.983 | 36.987 ± 2.615 | 43.852 ± 0.861 | 33.543 ± 0.464 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 11.545 ± 0.043 | 7.427 ± 0.173 | 7.652 ± 0.117 | 7.385 ± 0.045 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 77.053 ± 0.915 | 54.445 ± 0.691 | 53.129 ± 0.478 | 47.944 ± 2.061 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 7.707 ± 0.035 | 7.070 ± 0.089 | 8.019 ± 0.146 | 7.425 ± 0.033 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 51.088 ± 0.438 | 50.661 ± 0.952 | 52.993 ± 1.084 | 53.139 ± 2.325 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 2.205 ± 0.035 | 0.737 ± 0.010 | 1.267 ± 0.020 | 0.686 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 9.680 ± 0.191 | 4.274 ± 0.030 | 5.784 ± 0.060 | 4.209 ± 0.023 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 1.149 ± 0.025 | 0.741 ± 0.013 | 1.757 ± 0.039 | 0.685 ± 0.010 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 5.378 ± 0.128 | 4.272 ± 0.077 | 9.996 ± 0.134 | 4.184 ± 0.031 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 29.386 ± 0.200 | 15.192 ± 0.200 | 19.785 ± 0.187 | 15.881 ± 0.101 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 195.325 ± 1.119 | 103.222 ± 0.475 | 137.943 ± 1.605 | 107.444 ± 0.972 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 15.413 ± 0.255 | 15.070 ± 0.143 | 16.905 ± 0.272 | 15.994 ± 0.068 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 103.487 ± 1.585 | 104.584 ± 0.565 | 109.893 ± 0.734 | 114.760 ± 0.701 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.340 ± 0.003 | 0.034 ± 0.001 | 0.201 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.382 ± 0.043 | 0.040 ± 0.001 | 0.202 ± 0.001 | 0.014 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.366 ± 0.006 | 0.044 ± 0.001 | 0.209 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.201 ± 0.001 | 0.035 ± 0.001 | 0.197 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.215 ± 0.001 | 0.042 ± 0.001 | 0.200 ± 0.002 | 0.016 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.215 ± 0.002 | 0.047 ± 0.000 | 0.206 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.360 ± 0.011 | 0.032 ± 0.000 | 0.279 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.359 ± 0.007 | 0.034 ± 0.000 | 0.278 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.360 ± 0.004 | 0.034 ± 0.001 | 0.280 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.278 ± 0.108 | 0.032 ± 0.000 | 0.258 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.265 ± 0.012 | 0.032 ± 0.000 | 0.262 ± 0.006 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.265 ± 0.003 | 0.034 ± 0.000 | 0.261 ± 0.003 | 0.016 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.693 ± 0.010 | 0.057 ± 0.000 | 0.222 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.689 ± 0.017 | 0.059 ± 0.000 | 0.221 ± 0.007 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.694 ± 0.008 | 0.061 ± 0.001 | 0.225 ± 0.004 | 0.018 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.401 ± 0.002 | 0.060 ± 0.004 | 0.250 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.399 ± 0.010 | 0.063 ± 0.001 | 0.248 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.409 ± 0.003 | 0.065 ± 0.000 | 0.253 ± 0.006 | 0.015 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.281 ± 0.002 | 0.021 ± 0.000 | 0.411 ± 0.019 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.280 ± 0.002 | 0.021 ± 0.000 | 0.411 ± 0.007 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.278 ± 0.006 | 0.022 ± 0.000 | 0.413 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.192 ± 0.003 | 0.022 ± 0.001 | 0.247 ± 0.007 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.195 ± 0.006 | 0.021 ± 0.000 | 0.248 ± 0.003 | 0.015 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.196 ± 0.001 | 0.023 ± 0.001 | 0.252 ± 0.005 | 0.016 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.318 ± 0.010 | 0.022 ± 0.000 | 0.266 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.327 ± 0.003 | 0.027 ± 0.001 | 0.269 ± 0.006 | 0.017 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.359 ± 0.007 | 0.038 ± 0.001 | 0.288 ± 0.004 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.191 ± 0.004 | 0.021 ± 0.001 | 0.217 ± 0.006 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.198 ± 0.005 | 0.027 ± 0.001 | 0.221 ± 0.006 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.217 ± 0.007 | 0.038 ± 0.001 | 0.233 ± 0.003 | 0.020 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261104_213752/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261104_213752/run.yaml`
- Timestamp: `2026-11-04T21:37:52.180072+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261104_213752`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261104_213752/cpu_ops_t4_20261104_213752.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 44.000 ± 0.141 | 19.143 ± 2.553 | 7.265 ± 0.179 | 11.717 ± 0.185 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 88.928 ± 17.445 | 80.771 ± 22.199 | 43.520 ± 1.072 | 54.997 ± 3.868 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 25.256 ± 2.339 | 18.980 ± 0.952 | 6.019 ± 0.021 | 11.635 ± 0.870 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 58.486 ± 0.599 | 82.996 ± 31.621 | 31.571 ± 1.167 | 56.878 ± 9.461 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 37.286 ± 11.378 | 20.987 ± 8.555 | 9.204 ± 0.061 | 6.606 ± 1.025 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 64.865 ± 20.532 | 71.732 ± 21.638 | 48.945 ± 0.576 | 35.634 ± 2.008 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 35.987 ± 27.665 | 18.998 ± 6.260 | 3.825 ± 0.057 | 6.459 ± 0.807 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 92.784 ± 56.183 | 49.291 ± 19.043 | 24.901 ± 0.594 | 35.038 ± 6.568 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 41.740 ± 4.017 | 28.114 ± 27.640 | 5.134 ± 0.344 | 10.614 ± 1.487 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 103.003 ± 22.197 | 61.733 ± 38.504 | 34.392 ± 0.217 | 49.988 ± 7.309 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 35.787 ± 0.019 | 33.994 ± 9.934 | 5.405 ± 0.114 | 10.084 ± 0.737 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 85.668 ± 38.086 | 55.642 ± 0.832 | 33.613 ± 0.428 | 55.981 ± 11.336 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 6.746 ± 9.373 | 2.999 ± 0.020 | 2.288 ± 0.683 | 0.896 ± 0.119 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 36.143 ± 2.309 | 4.879 ± 3.409 | 3.059 ± 0.161 | 3.905 ± 0.093 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 3.340 ± 0.036 | 1.114 ± 0.020 | 1.307 ± 0.008 | 0.976 ± 0.033 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 10.754 ± 1.957 | 4.774 ± 0.817 | 4.771 ± 0.029 | 3.770 ± 0.021 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 45.712 ± 12.069 | 22.287 ± 0.110 | 17.255 ± 0.216 | 16.584 ± 0.991 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 170.738 ± 1.824 | 88.001 ± 1.646 | 95.288 ± 1.040 | 107.955 ± 4.710 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 34.794 ± 0.055 | 29.556 ± 2.748 | 15.387 ± 0.261 | 17.540 ± 1.230 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 98.334 ± 0.452 | 102.774 ± 1.230 | 83.910 ± 0.573 | 103.748 ± 7.075 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.378 ± 0.006 | 0.053 ± 0.016 | 0.199 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.389 ± 0.012 | 0.057 ± 0.000 | 0.204 ± 0.002 | 0.016 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.398 ± 0.019 | 0.062 ± 0.001 | 0.212 ± 0.004 | 0.019 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.230 ± 0.002 | 0.053 ± 0.026 | 0.199 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.408 ± 0.026 | 0.057 ± 0.001 | 0.204 ± 0.004 | 0.017 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.244 ± 0.006 | 0.061 ± 0.001 | 0.209 ± 0.004 | 0.020 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.400 ± 0.009 | 0.053 ± 0.001 | 0.384 ± 0.010 | 0.013 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.393 ± 0.002 | 0.053 ± 0.000 | 0.383 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.398 ± 0.004 | 0.054 ± 0.002 | 0.389 ± 0.012 | 0.021 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.294 ± 0.001 | 0.051 ± 0.000 | 0.341 ± 0.009 | 0.013 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.296 ± 0.096 | 0.051 ± 0.001 | 0.342 ± 0.007 | 0.015 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.296 ± 0.004 | 0.052 ± 0.002 | 0.344 ± 0.010 | 0.020 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.728 ± 0.007 | 0.086 ± 0.001 | 0.293 ± 0.008 | 0.014 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.736 ± 0.001 | 0.089 ± 0.000 | 0.286 ± 0.005 | 0.017 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.753 ± 0.004 | 0.090 ± 0.000 | 0.301 ± 0.011 | 0.022 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.438 ± 0.003 | 0.088 ± 0.002 | 0.319 ± 0.008 | 0.013 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.452 ± 0.007 | 0.090 ± 0.001 | 0.321 ± 0.008 | 0.017 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.449 ± 0.003 | 0.092 ± 0.003 | 0.335 ± 0.038 | 0.019 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.301 ± 0.010 | 0.034 ± 0.000 | 0.419 ± 0.011 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.306 ± 0.004 | 0.033 ± 0.000 | 0.416 ± 0.004 | 0.013 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.309 ± 0.008 | 0.034 ± 0.001 | 0.428 ± 0.010 | 0.014 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.218 ± 0.001 | 0.035 ± 0.001 | 0.252 ± 0.008 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.220 ± 0.002 | 0.035 ± 0.000 | 0.251 ± 0.003 | 0.015 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.222 ± 0.001 | 0.036 ± 0.000 | 0.257 ± 0.011 | 0.018 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.331 ± 0.002 | 0.032 ± 0.001 | 0.270 ± 0.002 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.355 ± 0.009 | 0.037 ± 0.000 | 0.272 ± 0.004 | 0.019 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.382 ± 0.005 | 0.049 ± 0.000 | 0.291 ± 0.004 | 0.026 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.210 ± 0.003 | 0.030 ± 0.000 | 0.220 ± 0.009 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.226 ± 0.009 | 0.036 ± 0.000 | 0.224 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.240 ± 0.003 | 0.048 ± 0.001 | 0.236 ± 0.004 | 0.033 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

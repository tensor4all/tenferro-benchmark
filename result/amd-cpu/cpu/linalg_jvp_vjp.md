# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `8eb4f087ca7cfae8e8f724cdb0b344ad5123e372`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261106_180658/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_180658/run.yaml`
- Timestamp: `2026-11-06T18:06:58.553919+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261106_180658`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261106_180658/cpu_ops_t1_20261106_180658.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 9.902 ± 0.171 | 5.028 ± 0.061 | 5.780 ± 0.084 | 7.255 ± 0.125 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 54.751 ± 1.173 | 28.770 ± 0.342 | 33.418 ± 0.994 | 43.982 ± 0.529 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 5.624 ± 0.067 | 4.991 ± 0.229 | 4.817 ± 0.132 | 7.268 ± 0.253 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 33.117 ± 0.823 | 28.982 ± 0.542 | 26.740 ± 0.643 | 43.969 ± 0.261 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.369 ± 0.352 | 2.239 ± 0.054 | 4.852 ± 0.148 | 5.124 ± 0.055 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 19.949 ± 0.684 | 16.007 ± 0.187 | 39.689 ± 0.512 | 33.342 ± 0.464 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.929 ± 0.042 | 2.391 ± 0.012 | 2.970 ± 0.074 | 5.114 ± 0.122 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 18.267 ± 0.149 | 16.699 ± 0.143 | 21.832 ± 0.420 | 34.148 ± 0.395 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 6.167 ± 0.076 | 3.495 ± 0.103 | 4.046 ± 0.121 | 6.642 ± 0.045 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 35.816 ± 0.828 | 24.312 ± 0.214 | 26.334 ± 0.325 | 39.192 ± 0.633 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.437 ± 0.036 | 3.600 ± 0.079 | 3.933 ± 0.074 | 6.614 ± 0.140 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 26.363 ± 0.440 | 25.086 ± 0.593 | 25.060 ± 0.208 | 45.017 ± 1.112 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.427 ± 0.021 | 0.387 ± 0.000 | 0.863 ± 0.026 | 0.593 ± 0.012 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 5.410 ± 0.146 | 2.193 ± 0.020 | 3.170 ± 0.043 | 3.584 ± 0.022 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.823 ± 0.018 | 0.397 ± 0.003 | 1.249 ± 0.079 | 0.736 ± 0.046 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.494 ± 0.189 | 2.301 ± 0.038 | 5.096 ± 0.165 | 3.898 ± 0.176 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 17.420 ± 0.320 | 8.417 ± 0.278 | 10.482 ± 0.308 | 12.827 ± 1.341 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 107.637 ± 1.788 | 55.459 ± 0.383 | 70.184 ± 1.517 | 76.563 ± 1.691 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 9.167 ± 0.085 | 8.627 ± 0.271 | 8.860 ± 0.363 | 12.279 ± 0.132 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 55.843 ± 0.448 | 56.122 ± 0.494 | 49.281 ± 0.680 | 84.567 ± 0.613 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.274 ± 0.007 | 0.019 ± 0.000 | 0.111 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.443 ± 0.066 | 0.021 ± 0.000 | 0.109 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.303 ± 0.005 | 0.026 ± 0.001 | 0.117 ± 0.006 | 0.021 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.122 ± 0.002 | 0.019 ± 0.000 | 0.100 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.129 ± 0.003 | 0.021 ± 0.000 | 0.103 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.136 ± 0.003 | 0.025 ± 0.001 | 0.108 ± 0.002 | 0.022 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.295 ± 0.003 | 0.023 ± 0.001 | 0.169 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.303 ± 0.006 | 0.023 ± 0.001 | 0.163 ± 0.004 | 0.012 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.310 ± 0.007 | 0.024 ± 0.001 | 0.167 ± 0.003 | 0.026 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.189 ± 0.006 | 0.022 ± 0.000 | 0.147 ± 0.008 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.203 ± 0.040 | 0.023 ± 0.001 | 0.143 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.196 ± 0.002 | 0.023 ± 0.000 | 0.144 ± 0.001 | 0.025 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.679 ± 0.007 | 0.034 ± 0.001 | 0.125 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.689 ± 0.008 | 0.034 ± 0.000 | 0.126 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.701 ± 0.021 | 0.035 ± 0.001 | 0.125 ± 0.002 | 0.023 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.347 ± 0.009 | 0.032 ± 0.002 | 0.133 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.358 ± 0.004 | 0.033 ± 0.001 | 0.134 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.369 ± 0.007 | 0.034 ± 0.001 | 0.135 ± 0.003 | 0.020 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.205 ± 0.006 | 0.015 ± 0.000 | 0.275 ± 0.005 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.212 ± 0.084 | 0.015 ± 0.000 | 0.277 ± 0.011 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.212 ± 0.005 | 0.016 ± 0.000 | 0.275 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.113 ± 0.003 | 0.015 ± 0.000 | 0.128 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.115 ± 0.002 | 0.015 ± 0.000 | 0.128 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.117 ± 0.003 | 0.015 ± 0.001 | 0.130 ± 0.002 | 0.013 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.241 ± 0.005 | 0.015 ± 0.000 | 0.155 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.260 ± 0.003 | 0.018 ± 0.000 | 0.160 ± 0.007 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.285 ± 0.010 | 0.027 ± 0.001 | 0.171 ± 0.004 | 0.027 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.114 ± 0.004 | 0.014 ± 0.001 | 0.112 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.124 ± 0.002 | 0.017 ± 0.001 | 0.115 ± 0.003 | 0.014 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.134 ± 0.000 | 0.027 ± 0.000 | 0.121 ± 0.001 | 0.028 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261106_184117/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_184117/run.yaml`
- Timestamp: `2026-11-06T18:41:17.633321+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261106_184117`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261106_184117/cpu_ops_t4_20261106_184117.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 25.956 ± 0.169 | 13.998 ± 0.035 | 4.778 ± 7.769 | 8.886 ± 0.844 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 53.513 ± 0.947 | 60.583 ± 11.208 | 23.314 ± 0.736 | 46.113 ± 1.272 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 17.199 ± 0.082 | 14.007 ± 1.008 | 4.069 ± 0.187 | 9.984 ± 2.054 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 65.749 ± 4.662 | 30.400 ± 2.304 | 17.432 ± 0.617 | 41.417 ± 9.239 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 23.094 ± 40.789 | 18.862 ± 11.204 | 3.568 ± 0.105 | 4.956 ± 1.502 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 60.824 ± 16.808 | 36.840 ± 18.204 | 23.519 ± 0.219 | 34.058 ± 8.580 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 16.837 ± 5.486 | 17.987 ± 19.697 | 2.577 ± 0.051 | 5.988 ± 2.767 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 27.493 ± 4.649 | 30.414 ± 50.109 | 12.821 ± 0.348 | 35.067 ± 5.841 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 69.003 ± 54.683 | 20.017 ± 0.049 | 3.127 ± 0.050 | 8.216 ± 1.951 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 46.957 ± 6.944 | 30.103 ± 41.253 | 15.124 ± 0.066 | 41.706 ± 3.227 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 20.816 ± 0.057 | 20.013 ± 0.998 | 3.043 ± 0.070 | 8.815 ± 3.252 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 30.396 ± 2.745 | 31.020 ± 25.827 | 14.573 ± 0.225 | 51.163 ± 9.736 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 8.628 ± 1.998 | 14.016 ± 4.940 | 0.878 ± 0.053 | 1.137 ± 0.436 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 18.192 ± 2.723 | 10.011 ± 9.003 | 2.118 ± 0.046 | 3.636 ± 0.536 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 6.156 ± 8.260 | 15.463 ± 2.962 | 1.008 ± 0.032 | 0.905 ± 0.058 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 16.650 ± 2.451 | 10.016 ± 5.993 | 3.151 ± 0.150 | 3.407 ± 0.157 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 46.799 ± 7.675 | 15.315 ± 0.248 | 9.903 ± 0.175 | 14.831 ± 1.213 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 102.443 ± 6.127 | 50.632 ± 6.455 | 51.274 ± 0.781 | 84.958 ± 5.077 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 20.854 ± 0.021 | 22.422 ± 3.049 | 8.608 ± 0.217 | 20.549 ± 3.851 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 61.993 ± 18.658 | 60.449 ± 4.065 | 38.924 ± 0.733 | 82.384 ± 13.050 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.292 ± 0.008 | 0.026 ± 0.000 | 0.108 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.309 ± 0.011 | 0.028 ± 0.001 | 0.112 ± 0.003 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.321 ± 0.007 | 0.033 ± 0.001 | 0.118 ± 0.005 | 0.019 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.136 ± 0.010 | 0.025 ± 0.001 | 0.101 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.139 ± 0.002 | 0.027 ± 0.001 | 0.102 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.148 ± 0.011 | 0.031 ± 0.001 | 0.108 ± 0.002 | 0.020 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.316 ± 0.011 | 0.032 ± 0.001 | 0.235 ± 0.004 | 0.011 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.319 ± 0.009 | 0.032 ± 0.001 | 0.233 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.327 ± 0.020 | 0.032 ± 0.001 | 0.234 ± 0.004 | 0.020 ± 0.002 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.208 ± 0.006 | 0.029 ± 0.001 | 0.182 ± 0.002 | 0.011 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.207 ± 0.003 | 0.030 ± 0.001 | 0.182 ± 0.004 | 0.012 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.212 ± 0.005 | 0.030 ± 0.000 | 0.183 ± 0.006 | 0.019 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.716 ± 0.005 | 0.045 ± 0.001 | 0.168 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.716 ± 0.002 | 0.045 ± 0.001 | 0.167 ± 0.005 | 0.012 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.726 ± 0.014 | 0.047 ± 0.002 | 0.169 ± 0.006 | 0.022 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.374 ± 0.009 | 0.044 ± 0.001 | 0.171 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.383 ± 0.011 | 0.044 ± 0.001 | 0.169 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.385 ± 0.014 | 0.046 ± 0.001 | 0.173 ± 0.004 | 0.020 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.219 ± 0.004 | 0.021 ± 0.001 | 0.271 ± 0.009 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.217 ± 0.005 | 0.021 ± 0.001 | 0.278 ± 0.017 | 0.011 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.227 ± 0.008 | 0.021 ± 0.001 | 0.272 ± 0.004 | 0.012 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.127 ± 0.002 | 0.020 ± 0.000 | 0.129 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.129 ± 0.005 | 0.020 ± 0.000 | 0.126 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.129 ± 0.001 | 0.021 ± 0.001 | 0.130 ± 0.004 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.252 ± 0.002 | 0.018 ± 0.000 | 0.159 ± 0.006 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.274 ± 0.014 | 0.021 ± 0.001 | 0.161 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.292 ± 0.005 | 0.031 ± 0.001 | 0.173 ± 0.005 | 0.025 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.121 ± 0.005 | 0.016 ± 0.000 | 0.111 ± 0.002 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.130 ± 0.006 | 0.020 ± 0.000 | 0.112 ± 0.001 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.143 ± 0.002 | 0.030 ± 0.001 | 0.122 ± 0.002 | 0.025 ± 0.001 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

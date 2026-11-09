# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261109_180601/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261109_180601/run.yaml`
- Timestamp: `2026-11-09T18:06:01.414278+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261109_180601`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261109_180601/cpu_ops_t1_20261109_180601.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 15.205 ± 0.061 | 9.926 ± 0.020 | 8.989 ± 0.060 | 7.291 ± 0.172 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 93.686 ± 0.893 | 64.956 ± 0.660 | 58.659 ± 0.257 | 45.596 ± 0.258 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 8.543 ± 0.175 | 9.931 ± 0.016 | 7.442 ± 0.289 | 7.209 ± 0.019 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 53.797 ± 0.789 | 66.082 ± 0.441 | 45.828 ± 0.366 | 45.242 ± 0.343 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 9.795 ± 0.102 | 4.716 ± 0.020 | 11.502 ± 0.284 | 4.216 ± 0.053 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 65.053 ± 0.538 | 34.675 ± 0.899 | 72.355 ± 0.639 | 28.499 ± 0.394 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 8.946 ± 0.044 | 4.991 ± 0.073 | 5.459 ± 0.124 | 4.097 ± 0.028 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 61.951 ± 0.272 | 35.967 ± 0.335 | 39.191 ± 0.816 | 28.428 ± 0.360 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 15.264 ± 0.254 | 8.841 ± 0.199 | 6.412 ± 0.067 | 5.795 ± 0.046 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 103.726 ± 0.523 | 66.483 ± 0.176 | 46.951 ± 0.773 | 37.755 ± 0.135 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 11.175 ± 0.139 | 8.563 ± 0.073 | 6.268 ± 0.054 | 5.728 ± 0.181 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 81.657 ± 0.414 | 62.411 ± 0.804 | 46.359 ± 0.483 | 41.700 ± 0.218 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 1.701 ± 0.024 | 1.092 ± 0.012 | 1.025 ± 0.027 | 0.566 ± 0.004 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 7.851 ± 0.191 | 6.927 ± 0.119 | 5.043 ± 0.348 | 3.560 ± 0.050 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.756 ± 0.005 | 1.099 ± 0.003 | 1.829 ± 0.042 | 0.596 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 4.131 ± 0.362 | 6.894 ± 0.076 | 8.225 ± 0.245 | 3.648 ± 0.038 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 25.003 ± 0.440 | 16.893 ± 0.076 | 18.885 ± 0.395 | 12.495 ± 0.098 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 168.108 ± 1.758 | 121.214 ± 0.335 | 119.197 ± 2.673 | 88.865 ± 0.726 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 13.085 ± 0.192 | 16.769 ± 0.102 | 13.814 ± 0.159 | 12.543 ± 0.066 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 88.422 ± 0.756 | 122.017 ± 0.550 | 95.508 ± 0.323 | 94.159 ± 1.959 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.191 ± 0.002 | 0.022 ± 0.000 | 0.119 ± 0.007 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.204 ± 0.009 | 0.025 ± 0.000 | 0.124 ± 0.001 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.212 ± 0.005 | 0.030 ± 0.000 | 0.129 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.060 ± 0.001 | 0.022 ± 0.000 | 0.114 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.071 ± 0.001 | 0.025 ± 0.000 | 0.120 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.073 ± 0.002 | 0.030 ± 0.000 | 0.123 ± 0.001 | 0.012 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.568 ± 0.006 | 0.027 ± 0.000 | 0.180 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.576 ± 0.004 | 0.027 ± 0.000 | 0.178 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.577 ± 0.008 | 0.028 ± 0.000 | 0.180 ± 0.003 | 0.014 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.139 ± 0.001 | 0.025 ± 0.000 | 0.165 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.143 ± 0.002 | 0.025 ± 0.000 | 0.167 ± 0.006 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.146 ± 0.005 | 0.026 ± 0.000 | 0.166 ± 0.004 | 0.016 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.760 ± 0.006 | 0.037 ± 0.000 | 0.139 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.786 ± 0.005 | 0.039 ± 0.000 | 0.138 ± 0.002 | 0.010 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.809 ± 0.010 | 0.041 ± 0.001 | 0.139 ± 0.004 | 0.014 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.137 ± 0.001 | 0.038 ± 0.000 | 0.159 ± 0.005 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.139 ± 0.003 | 0.039 ± 0.000 | 0.158 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.143 ± 0.001 | 0.042 ± 0.001 | 0.161 ± 0.003 | 0.012 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.140 ± 0.001 | 0.020 ± 0.000 | 0.267 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.141 ± 0.001 | 0.021 ± 0.000 | 0.271 ± 0.004 | 0.014 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.143 ± 0.001 | 0.021 ± 0.000 | 0.274 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.061 ± 0.001 | 0.020 ± 0.000 | 0.150 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.061 ± 0.001 | 0.020 ± 0.000 | 0.151 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.063 ± 0.001 | 0.021 ± 0.000 | 0.151 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.165 ± 0.001 | 0.016 ± 0.000 | 0.177 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.181 ± 0.005 | 0.020 ± 0.000 | 0.183 ± 0.003 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.197 ± 0.003 | 0.031 ± 0.000 | 0.195 ± 0.004 | 0.015 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.056 ± 0.002 | 0.015 ± 0.000 | 0.133 ± 0.001 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.062 ± 0.001 | 0.019 ± 0.000 | 0.140 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.074 ± 0.001 | 0.030 ± 0.000 | 0.149 ± 0.004 | 0.015 ± 0.003 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261109_184059/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261109_184059/run.yaml`
- Timestamp: `2026-11-09T18:40:59.908465+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/amd-cpu/cpu/einsum/20261109_184059`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261109_184059/cpu_ops_t4_20261109_184059.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 27.866 ± 0.164 | 21.998 ± 0.073 | 6.173 ± 1.671 | 8.985 ± 2.795 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 129.677 ± 0.401 | 97.993 ± 0.368 | 35.267 ± 0.592 | 46.431 ± 5.941 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 13.754 ± 0.073 | 22.008 ± 0.077 | 5.073 ± 0.144 | 7.722 ± 1.059 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 71.309 ± 0.368 | 98.685 ± 0.281 | 25.151 ± 0.636 | 44.984 ± 2.629 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 14.817 ± 0.299 | 8.234 ± 0.070 | 6.876 ± 0.173 | 4.938 ± 2.156 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 290.628 ± 27.039 | 233.943 ± 1.695 | 40.443 ± 0.662 | 30.984 ± 2.508 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 12.116 ± 0.126 | 8.393 ± 0.075 | 3.101 ± 0.039 | 5.400 ± 1.059 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 186.132 ± 20.673 | 196.094 ± 33.170 | 20.969 ± 0.492 | 28.495 ± 1.026 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 95.773 ± 56.236 | 13.308 ± 0.068 | 4.152 ± 0.042 | 7.094 ± 1.649 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 404.433 ± 129.970 | 377.646 ± 3.690 | 29.303 ± 0.252 | 46.986 ± 4.026 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 14.899 ± 0.078 | 13.052 ± 0.205 | 4.668 ± 0.048 | 8.990 ± 2.964 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 246.302 ± 2.705 | 358.346 ± 17.699 | 28.230 ± 0.328 | 44.959 ± 7.982 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 3.684 ± 0.030 | 2.732 ± 0.010 | 1.802 ± 1.800 | 0.715 ± 0.017 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 214.267 ± 15.102 | 188.976 ± 36.012 | 2.547 ± 0.127 | 3.172 ± 0.031 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 1.724 ± 0.021 | 2.747 ± 0.023 | 1.017 ± 0.030 | 0.821 ± 0.059 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 123.609 ± 20.781 | 202.040 ± 10.972 | 4.016 ± 0.027 | 3.230 ± 0.050 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 31.975 ± 0.272 | 21.405 ± 0.114 | 16.310 ± 0.239 | 14.241 ± 0.997 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 204.727 ± 1.498 | 151.267 ± 0.476 | 81.583 ± 1.964 | 84.970 ± 6.991 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 17.576 ± 0.062 | 23.011 ± 0.291 | 12.982 ± 0.113 | 13.733 ± 0.568 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 108.217 ± 0.072 | 152.196 ± 0.091 | 71.072 ± 0.691 | 86.764 ± 6.325 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.202 ± 0.004 | 0.030 ± 0.000 | 0.120 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.216 ± 0.004 | 0.034 ± 0.000 | 0.125 ± 0.007 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.224 ± 0.002 | 0.039 ± 0.000 | 0.131 ± 0.001 | 0.013 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.072 ± 0.001 | 0.030 ± 0.000 | 0.115 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.080 ± 0.001 | 0.034 ± 0.000 | 0.121 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.086 ± 0.001 | 0.039 ± 0.000 | 0.125 ± 0.002 | 0.013 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.585 ± 0.008 | 0.037 ± 0.000 | 0.248 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.605 ± 0.043 | 0.038 ± 0.000 | 0.243 ± 0.003 | 0.010 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.600 ± 0.003 | 0.038 ± 0.000 | 0.245 ± 0.005 | 0.017 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.172 ± 0.003 | 0.033 ± 0.000 | 0.211 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.173 ± 0.003 | 0.034 ± 0.000 | 0.211 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.179 ± 0.003 | 0.035 ± 0.000 | 0.212 ± 0.002 | 0.015 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.778 ± 0.009 | 0.052 ± 0.001 | 0.179 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.785 ± 0.005 | 0.054 ± 0.000 | 0.177 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.803 ± 0.013 | 0.056 ± 0.001 | 0.180 ± 0.004 | 0.016 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.166 ± 0.001 | 0.053 ± 0.001 | 0.195 ± 0.003 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.168 ± 0.004 | 0.054 ± 0.000 | 0.197 ± 0.004 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.172 ± 0.001 | 0.057 ± 0.000 | 0.197 ± 0.004 | 0.014 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.148 ± 0.001 | 0.028 ± 0.000 | 0.276 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.149 ± 0.001 | 0.028 ± 0.000 | 0.275 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.156 ± 0.002 | 0.029 ± 0.000 | 0.277 ± 0.002 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.068 ± 0.001 | 0.027 ± 0.000 | 0.152 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.068 ± 0.001 | 0.028 ± 0.000 | 0.154 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.071 ± 0.003 | 0.029 ± 0.000 | 0.153 ± 0.001 | 0.011 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.174 ± 0.002 | 0.021 ± 0.000 | 0.179 ± 0.034 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.187 ± 0.001 | 0.025 ± 0.000 | 0.187 ± 0.006 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.208 ± 0.001 | 0.036 ± 0.000 | 0.199 ± 0.005 | 0.017 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.063 ± 0.000 | 0.019 ± 0.000 | 0.134 ± 0.006 | 0.009 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.072 ± 0.002 | 0.024 ± 0.000 | 0.139 ± 0.002 | 0.012 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.080 ± 0.001 | 0.035 ± 0.000 | 0.149 ± 0.001 | 0.019 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`
- Benchmark commit: `1711cc92c7d04a38b85df1170873ddd4cf029b5b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261022_022731/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261022_022731`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261022_022731`.

- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_022731/cpu_ops_t1_20261022_022731.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_022731/linalg_jvp_vjp_t1_20261022_022731.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 4.735 ± 0.148 | 4.491 ± 0.066 | 5.690 ± 0.045 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 23.659 ± 0.172 | 25.596 ± 0.311 | 34.192 ± 0.146 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 4.806 ± 0.072 | 4.185 ± 0.008 | 5.687 ± 0.027 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 23.359 ± 0.041 | 23.217 ± 0.050 | 34.422 ± 5.497 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 0.906 ± 0.004 | 1.622 ± 0.017 | 2.360 ± 0.005 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 5.477 ± 0.030 | 12.007 ± 0.399 | 20.198 ± 1.958 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 1.006 ± 0.018 | 0.992 ± 0.002 | 2.233 ± 0.011 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 5.967 ± 0.042 | 6.809 ± 0.322 | 18.438 ± 2.940 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 1.816 ± 0.003 | 2.173 ± 0.122 | 3.933 ± 0.045 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 12.142 ± 0.149 | 12.242 ± 0.115 | 25.016 ± 0.190 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 1.856 ± 0.006 | 2.047 ± 0.009 | 3.832 ± 0.030 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 10.586 ± 0.849 | 11.682 ± 0.225 | 25.224 ± 0.120 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.260 ± 0.010 | 0.380 ± 0.029 | 0.252 ± 0.001 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 1.156 ± 0.010 | 1.422 ± 0.022 | 1.271 ± 0.007 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.263 ± 0.005 | 0.465 ± 0.035 | 0.244 ± 0.001 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 1.137 ± 0.011 | 2.376 ± 0.099 | 1.328 ± 0.087 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 7.858 ± 0.022 | 10.236 ± 0.946 | 9.361 ± 0.086 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 45.687 ± 7.019 | 49.585 ± 2.594 | 53.827 ± 0.448 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 7.857 ± 0.034 | 7.819 ± 0.051 | 9.453 ± 0.127 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 54.437 ± 7.990 | 43.228 ± 0.172 | 54.086 ± 0.155 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.015 ± 0.000 | 0.085 ± 0.045 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.061 ± 0.002 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.023 ± 0.000 | 0.070 ± 0.001 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.014 ± 0.000 | 0.053 ± 0.007 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.053 ± 0.002 | 0.008 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.019 ± 0.000 | 0.062 ± 0.002 | 0.009 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.015 ± 0.000 | 0.092 ± 0.027 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.016 ± 0.000 | 0.089 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.016 ± 0.000 | 0.096 ± 0.000 | 0.004 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.015 ± 0.000 | 0.072 ± 0.005 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.015 ± 0.000 | 0.071 ± 0.000 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.019 ± 0.000 | 0.078 ± 0.001 | 0.005 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.025 ± 0.000 | 0.067 ± 0.002 | 0.006 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.025 ± 0.000 | 0.069 ± 0.003 | 0.007 ± 0.000 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.026 ± 0.000 | 0.074 ± 0.001 | 0.005 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.025 ± 0.000 | 0.086 ± 0.152 | 0.006 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.025 ± 0.000 | 0.070 ± 0.004 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.030 ± 0.000 | 0.074 ± 0.000 | 0.005 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.011 ± 0.000 | 0.150 ± 0.015 | 0.007 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.142 ± 0.001 | 0.006 ± 0.000 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.011 ± 0.000 | 0.142 ± 0.003 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.011 ± 0.000 | 0.069 ± 0.002 | 0.009 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.011 ± 0.000 | 0.068 ± 0.000 | 0.006 ± 0.000 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.011 ± 0.000 | 0.069 ± 0.001 | 0.007 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.010 ± 0.000 | 0.084 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.012 ± 0.000 | 0.087 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.022 ± 0.000 | 0.097 ± 0.000 | 0.018 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.011 ± 0.000 | 0.057 ± 0.003 | 0.008 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.013 ± 0.000 | 0.058 ± 0.001 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.020 ± 0.000 | 0.065 ± 0.000 | 0.018 ± 0.000 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261022_025208/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20261022_025208`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261022_025208`.

- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261022_025208/cpu_ops_t4_20261022_025208.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20261022_025208/linalg_jvp_vjp_t4_20261022_025208.md`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 11.961 ± 7.921 | 10.756 ± 4.283 | 7.699 ± 3.331 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 32.995 ± 2.375 | 48.157 ± 16.802 | 41.825 ± 2.244 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 10.691 ± 7.135 | 7.650 ± 3.724 | 6.704 ± 0.273 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 31.090 ± 3.588 | 29.954 ± 1.181 | 39.845 ± 1.771 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.975 ± 7.913 | 7.496 ± 1.791 | 3.065 ± 1.108 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 8.326 ± 0.667 | 17.710 ± 4.739 | 18.994 ± 3.388 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 1.836 ± 0.704 | 4.443 ± 2.317 | 2.518 ± 0.459 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 9.093 ± 0.815 | 9.387 ± 2.003 | 14.493 ± 1.373 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.024 ± 2.334 | 8.126 ± 3.303 | 6.362 ± 2.898 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 15.086 ± 1.052 | 23.689 ± 8.305 | 14.586 ± 0.777 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.928 ± 3.243 | 7.859 ± 1.700 | 8.090 ± 2.111 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 15.054 ± 0.892 | 22.992 ± 4.373 | 26.205 ± 12.592 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.626 ± 0.196 | 1.488 ± 0.917 | 0.939 ± 0.853 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.644 ± 0.163 | 1.927 ± 0.323 | 1.709 ± 0.239 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.483 ± 0.814 | 1.374 ± 1.133 | 0.583 ± 0.427 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.492 ± 0.085 | 3.443 ± 2.024 | 1.766 ± 0.415 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 17.255 ± 6.152 | 15.140 ± 5.916 | 14.644 ± 4.862 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 79.522 ± 11.976 | 96.707 ± 9.882 | 62.320 ± 10.872 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 11.664 ± 5.591 | 12.127 ± 4.054 | 15.354 ± 4.508 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 62.277 ± 3.115 | 76.158 ± 8.007 | 58.455 ± 6.454 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.018 ± 0.000 | 0.070 ± 0.007 | 0.009 ± 0.004 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.019 ± 0.000 | 0.100 ± 0.098 | 0.012 ± 0.003 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.027 ± 0.000 | 0.086 ± 0.072 | 0.010 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.019 ± 0.001 | 0.062 ± 0.009 | 0.010 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.018 ± 0.001 | 0.069 ± 0.043 | 0.013 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.024 ± 0.002 | 0.107 ± 0.025 | 0.009 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.019 ± 0.001 | 0.661 ± 0.071 | 0.007 ± 0.002 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.000 | 0.546 ± 0.174 | 0.009 ± 0.001 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.019 ± 0.002 | 0.614 ± 0.225 | 0.017 ± 0.006 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.019 ± 0.002 | 0.522 ± 0.167 | 0.011 ± 0.007 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.017 ± 0.001 | 0.484 ± 0.253 | 0.012 ± 0.003 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.025 ± 0.005 | 0.513 ± 0.109 | 0.015 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.030 ± 0.002 | 0.368 ± 0.147 | 0.013 ± 0.015 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.032 ± 0.005 | 0.177 ± 0.091 | 0.012 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.031 ± 0.002 | 0.229 ± 0.151 | 0.007 ± 0.003 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.028 ± 0.005 | 0.191 ± 0.082 | 0.007 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.036 ± 0.010 | 0.237 ± 0.188 | 0.012 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.035 ± 0.003 | 0.202 ± 0.107 | 0.012 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.013 ± 0.000 | 0.166 ± 0.052 | 0.011 ± 0.005 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.013 ± 0.000 | 0.266 ± 0.122 | 0.007 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.013 ± 0.003 | 0.171 ± 0.024 | 0.008 ± 0.005 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.014 ± 0.003 | 0.083 ± 0.048 | 0.009 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.013 ± 0.001 | 0.105 ± 0.137 | 0.008 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.014 ± 0.001 | 0.078 ± 0.006 | 0.010 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.014 ± 0.003 | 0.135 ± 0.108 | 0.011 ± 0.003 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.014 ± 0.003 | 0.108 ± 0.054 | 0.015 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.026 ± 0.001 | 0.124 ± 0.010 | 0.018 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.013 ± 0.001 | 0.069 ± 0.045 | 0.011 ± 0.003 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.015 ± 0.001 | 0.071 ± 0.003 | 0.014 ± 0.012 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.024 ± 0.000 | 0.081 ± 0.017 | 0.022 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

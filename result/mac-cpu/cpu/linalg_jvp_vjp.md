# CPU Linalg JVP/VJP Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261109_172454/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_172454/run.yaml`
- Timestamp: `2026-11-09T17:24:54.736713+00:00`

Latest run: `./scripts/run_all.sh 1`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261109_172454`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261109_172454/cpu_ops_t1_20261109_172454.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | 12.982 ± 2.296 | 8.334 ± 0.546 | 5.514 ± 0.311 | 7.176 ± 0.602 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | 72.339 ± 2.928 | 48.127 ± 7.627 | 33.598 ± 1.574 | 66.609 ± 22.296 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | 6.880 ± 0.669 | 8.052 ± 0.198 | 5.374 ± 0.254 | 7.044 ± 0.764 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | 34.480 ± 7.485 | 57.444 ± 0.641 | 29.813 ± 1.951 | 53.011 ± 5.390 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | 3.095 ± 0.181 | 1.290 ± 0.055 | 2.148 ± 0.093 | 3.999 ± 1.453 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | 18.616 ± 1.717 | 9.642 ± 0.757 | 16.246 ± 0.680 | 24.832 ± 0.866 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | 2.964 ± 1.362 | 1.306 ± 0.027 | 1.337 ± 0.233 | 2.988 ± 0.380 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | 16.741 ± 4.316 | 11.052 ± 4.757 | 8.537 ± 0.169 | 25.766 ± 2.131 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | 11.088 ± 2.727 | 3.811 ± 0.153 | 2.843 ± 0.248 | 5.517 ± 1.022 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | 31.207 ± 4.834 | 22.104 ± 3.651 | 17.563 ± 0.332 | 41.720 ± 6.274 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | 4.317 ± 1.156 | 3.900 ± 0.094 | 2.960 ± 0.357 | 4.782 ± 0.270 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | 29.250 ± 14.078 | 20.281 ± 0.789 | 17.015 ± 0.822 | 34.395 ± 2.210 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | 0.787 ± 0.172 | 0.637 ± 0.074 | 0.471 ± 0.055 | 0.367 ± 0.070 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | 3.704 ± 0.568 | 4.624 ± 1.268 | 2.057 ± 0.093 | 3.464 ± 1.219 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | 0.455 ± 0.114 | 0.547 ± 0.026 | 0.638 ± 0.097 | 0.388 ± 0.040 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | 3.605 ± 1.220 | 3.493 ± 1.054 | 3.206 ± 0.370 | 2.506 ± 0.644 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | 29.484 ± 3.290 | 13.871 ± 0.712 | 10.587 ± 0.391 | 12.164 ± 1.152 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | 110.877 ± 4.949 | 89.552 ± 5.473 | 64.984 ± 3.012 | 94.908 ± 7.111 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | 14.803 ± 10.388 | 14.999 ± 2.856 | 9.686 ± 0.080 | 11.032 ± 0.255 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | 55.349 ± 3.779 | 103.588 ± 22.155 | 56.117 ± 1.629 | 85.119 ± 5.348 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | 0.113 ± 0.005 | 0.015 ± 0.001 | 0.065 ± 0.004 | 0.007 ± 0.004 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | 0.102 ± 0.002 | 0.022 ± 0.003 | 0.070 ± 0.004 | 0.009 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | 0.130 ± 0.010 | 0.029 ± 0.004 | 0.121 ± 0.043 | 0.016 ± 0.004 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | 0.039 ± 0.003 | 0.016 ± 0.001 | 0.103 ± 0.022 | 0.008 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | 0.034 ± 0.001 | 0.021 ± 0.002 | 0.087 ± 0.016 | 0.010 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | 0.056 ± 0.033 | 0.025 ± 0.004 | 0.097 ± 0.016 | 0.017 ± 0.011 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | 0.517 ± 0.145 | 0.018 ± 0.000 | 0.131 ± 0.016 | 0.010 ± 0.004 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | 0.344 ± 0.007 | 0.021 ± 0.000 | 0.156 ± 0.019 | 0.008 ± 0.003 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | 0.442 ± 0.115 | 0.019 ± 0.002 | 0.142 ± 0.018 | 0.015 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | 0.094 ± 0.008 | 0.017 ± 0.003 | 0.117 ± 0.010 | 0.008 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | 0.077 ± 0.003 | 0.019 ± 0.000 | 0.109 ± 0.054 | 0.010 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | 0.167 ± 0.073 | 0.023 ± 0.001 | 0.121 ± 0.015 | 0.019 ± 0.009 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | 0.602 ± 0.029 | 0.028 ± 0.002 | 0.096 ± 0.006 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | 0.476 ± 0.024 | 0.033 ± 0.002 | 0.112 ± 0.026 | 0.010 ± 0.002 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | 0.513 ± 0.012 | 0.035 ± 0.003 | 0.120 ± 0.019 | 0.009 ± 0.002 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | 0.072 ± 0.003 | 0.028 ± 0.001 | 0.111 ± 0.028 | 0.009 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | 0.069 ± 0.003 | 0.032 ± 0.005 | 0.134 ± 0.034 | 0.010 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | 0.081 ± 0.003 | 0.041 ± 0.001 | 0.108 ± 0.005 | 0.022 ± 0.015 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | 0.086 ± 0.006 | 0.014 ± 0.002 | 0.204 ± 0.028 | 0.010 ± 0.003 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | 0.069 ± 0.004 | 0.017 ± 0.003 | 0.232 ± 0.048 | 0.009 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | 0.079 ± 0.006 | 0.018 ± 0.001 | 0.234 ± 0.061 | 0.010 ± 0.006 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | 0.042 ± 0.002 | 0.014 ± 0.000 | 0.096 ± 0.007 | 0.009 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | 0.034 ± 0.002 | 0.015 ± 0.001 | 0.105 ± 0.014 | 0.010 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | 0.043 ± 0.012 | 0.017 ± 0.002 | 0.109 ± 0.026 | 0.011 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | 0.115 ± 0.005 | 0.012 ± 0.000 | 0.173 ± 0.102 | 0.012 ± 0.002 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | 0.097 ± 0.005 | 0.017 ± 0.000 | 0.136 ± 0.033 | 0.013 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | 0.115 ± 0.005 | 0.031 ± 0.001 | 0.144 ± 0.036 | 0.021 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | 0.034 ± 0.004 | 0.011 ± 0.000 | 0.104 ± 0.028 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | 0.036 ± 0.002 | 0.017 ± 0.001 | 0.088 ± 0.013 | 0.014 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | 0.041 ± 0.002 | 0.029 ± 0.003 | 0.114 ± 0.018 | 0.021 ± 0.002 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261109_180327/linalg_jvp_vjp_report.md`.


- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261109_180327/run.yaml`
- Timestamp: `2026-11-09T18:03:27.099105+00:00`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20261109_180327`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261109_180327/cpu_ops_t4_20261109_180327.csv`

### Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs measures both arms: eager `EagerRuntime::jvp` / `vjp`, and trace-mode
`AdContext`. PyTorch uses `torch.func.jvp` / `vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 12.108 ± 1.404 | 10.162 ± 2.745 | 4.919 ± 0.054 | 5.297 ± 0.049 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 62.771 ± 57.417 | 63.180 ± 9.878 | 27.829 ± 2.735 | 28.352 ± 0.336 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 5.844 ± 0.792 | 10.545 ± 2.332 | 5.113 ± 0.178 | 5.354 ± 0.051 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 31.163 ± 1.770 | 66.406 ± 14.136 | 25.898 ± 2.617 | 28.525 ± 0.133 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 3.262 ± 0.212 | 1.388 ± 1.128 | 2.250 ± 0.479 | 1.476 ± 0.046 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 18.164 ± 5.191 | 11.494 ± 2.289 | 17.941 ± 12.105 | 8.962 ± 0.366 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.889 ± 0.262 | 1.590 ± 0.800 | 1.526 ± 0.091 | 1.423 ± 0.034 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 19.329 ± 0.867 | 14.992 ± 1.610 | 8.157 ± 2.508 | 8.918 ± 0.367 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 7.466 ± 0.958 | 3.835 ± 0.142 | 3.617 ± 0.169 | 2.892 ± 0.075 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 32.844 ± 3.642 | 34.171 ± 9.727 | 20.780 ± 2.812 | 15.161 ± 0.278 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 5.115 ± 0.618 | 5.244 ± 4.545 | 3.530 ± 0.360 | 2.934 ± 0.124 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 24.535 ± 2.810 | 21.388 ± 9.973 | 23.658 ± 14.888 | 15.780 ± 0.433 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.931 ± 0.227 | 0.578 ± 0.089 | 0.533 ± 0.031 | 0.352 ± 0.010 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 3.707 ± 1.198 | 4.306 ± 1.353 | 1.881 ± 0.716 | 1.712 ± 0.212 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.340 ± 0.056 | 0.717 ± 0.096 | 0.839 ± 0.102 | 0.344 ± 0.006 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 2.719 ± 0.707 | 3.906 ± 0.895 | 2.715 ± 0.186 | 1.440 ± 0.053 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 26.071 ± 2.665 | 15.719 ± 1.646 | 9.641 ± 0.480 | 10.342 ± 0.202 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 116.366 ± 21.807 | 109.638 ± 13.460 | 54.710 ± 3.213 | 52.184 ± 2.700 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 14.732 ± 1.798 | 17.687 ± 1.083 | 10.574 ± 2.592 | 10.315 ± 0.230 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 70.834 ± 16.299 | 90.015 ± 6.905 | 50.786 ± 5.449 | 53.870 ± 1.717 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.134 ± 0.014 | 0.025 ± 0.003 | 0.082 ± 0.015 | 0.006 ± 0.000 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.114 ± 0.006 | 0.026 ± 0.001 | 0.066 ± 0.004 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.133 ± 0.013 | 0.033 ± 0.001 | 0.093 ± 0.010 | 0.010 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.043 ± 0.001 | 0.023 ± 0.001 | 0.129 ± 0.060 | 0.007 ± 0.000 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.047 ± 0.008 | 0.034 ± 0.020 | 0.058 ± 0.002 | 0.009 ± 0.001 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.056 ± 0.003 | 0.030 ± 0.004 | 0.076 ± 0.006 | 0.012 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.417 ± 0.033 | 0.028 ± 0.001 | 0.402 ± 0.121 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.443 ± 0.054 | 0.029 ± 0.001 | 0.282 ± 0.011 | 0.007 ± 0.000 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.385 ± 0.009 | 0.029 ± 0.006 | 0.309 ± 0.030 | 0.009 ± 0.008 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.134 ± 0.014 | 0.026 ± 0.003 | 0.281 ± 0.214 | 0.007 ± 0.000 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.108 ± 0.008 | 0.025 ± 0.002 | 0.230 ± 0.040 | 0.007 ± 0.001 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.119 ± 0.006 | 0.035 ± 0.003 | 0.259 ± 0.007 | 0.006 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.562 ± 0.310 | 0.042 ± 0.002 | 0.150 ± 0.017 | 0.008 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.592 ± 0.303 | 0.043 ± 0.002 | 0.218 ± 0.019 | 0.009 ± 0.001 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.525 ± 0.023 | 0.050 ± 0.039 | 0.193 ± 0.059 | 0.005 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.097 ± 0.036 | 0.043 ± 0.002 | 0.179 ± 0.029 | 0.007 ± 0.001 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.102 ± 0.029 | 0.042 ± 0.002 | 0.226 ± 0.025 | 0.009 ± 0.000 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.120 ± 0.010 | 0.048 ± 0.004 | 0.197 ± 0.029 | 0.018 ± 0.006 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.114 ± 0.045 | 0.022 ± 0.002 | 0.183 ± 0.129 | 0.007 ± 0.001 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.087 ± 0.013 | 0.021 ± 0.001 | 0.176 ± 0.028 | 0.007 ± 0.002 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.096 ± 0.022 | 0.028 ± 0.010 | 0.190 ± 0.025 | 0.007 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.063 ± 0.011 | 0.023 ± 0.007 | 0.095 ± 0.037 | 0.007 ± 0.001 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.049 ± 0.005 | 0.021 ± 0.001 | 0.105 ± 0.023 | 0.008 ± 0.002 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.045 ± 0.007 | 0.022 ± 0.001 | 0.087 ± 0.005 | 0.008 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.178 ± 0.152 | 0.018 ± 0.001 | 0.093 ± 0.005 | 0.010 ± 0.000 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.120 ± 0.017 | 0.021 ± 0.002 | 0.116 ± 0.022 | 0.012 ± 0.001 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.125 ± 0.028 | 0.037 ± 0.003 | 0.134 ± 0.018 | 0.035 ± 0.026 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.069 ± 0.026 | 0.016 ± 0.002 | 0.064 ± 0.013 | 0.011 ± 0.001 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.041 ± 0.001 | 0.022 ± 0.001 | 0.086 ± 0.021 | 0.013 ± 0.002 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.051 ± 0.004 | 0.033 ± 0.005 | 0.092 ± 0.010 | 0.023 ± 0.006 |

### Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

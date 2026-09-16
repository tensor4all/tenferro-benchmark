# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`
- Benchmark commit: `f983e88b9d07e558989427d047f1fa4676df2ea8`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_102753/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260916_102753`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260916_102753`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_102753/cpu_ops_t1_20260916_102753.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_102753/cpu_ops_t1_20260916_102753.md`

### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 10.910 ± 0.074 | 11.040 ± 0.128 | 9.914 ± 0.096 | 10.404 ± 0.084 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.178 ± 0.003 | 0.176 ± 0.001 | 0.165 ± 0.014 | 0.188 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.764 ± 0.038 | 2.768 ± 0.048 | 2.511 ± 0.074 | 2.617 ± 0.101 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.691 ± 0.021 | 0.692 ± 0.012 | 0.630 ± 0.031 | 0.660 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.966 ± 0.013 | 0.957 ± 0.027 | 0.131 ± 0.005 | 0.299 ± 0.022 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.234 ± 0.011 | 0.242 ± 0.009 | 0.036 ± 0.003 | 0.089 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.066 ± 0.007 | 0.066 ± 0.001 | 0.011 ± 0.005 | 0.038 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.575 ± 0.030 | 1.535 ± 0.056 | 0.686 ± 0.014 | 0.858 ± 0.053 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.384 ± 0.008 | 0.388 ± 0.009 | 0.174 ± 0.003 | 0.231 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.102 ± 0.004 | 0.101 ± 0.005 | 0.045 ± 0.003 | 0.075 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 3.521 ± 0.090 | 3.470 ± 0.077 | 2.583 ± 0.101 | 2.738 ± 0.071 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.059 ± 0.000 | 0.061 ± 0.000 | 0.044 ± 0.001 | 0.065 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.858 ± 0.053 | 0.884 ± 0.026 | 0.646 ± 0.017 | 0.731 ± 0.043 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.221 ± 0.010 | 0.223 ± 0.002 | 0.164 ± 0.003 | 0.195 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.289 ± 0.023 | 0.294 ± 0.019 | 1.045 ± 0.043 | 0.389 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.020 ± 0.001 | 0.078 ± 0.016 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.077 ± 0.000 | 0.080 ± 0.000 | 0.276 ± 0.027 | 0.188 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.025 ± 0.000 | 0.030 ± 0.002 | 0.064 ± 0.004 | 0.102 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.037 ± 0.000 | 0.039 ± 0.000 | 0.009 ± 0.001 | 0.077 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.005 ± 0.001 | 0.052 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.000 | 0.063 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.001 | 0.066 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.002 | 0.054 ± 0.003 | 0.022 ± 0.004 | 0.116 ± 0.059 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.057 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.001 | 0.072 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.065 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.114 ± 0.001 | 0.108 ± 0.001 | 0.500 ± 0.022 | 0.233 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.066 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.032 ± 0.000 | 0.035 ± 0.001 | 0.126 ± 0.007 | 0.091 ± 0.044 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.035 ± 0.001 | 0.088 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 2.900 ± 0.090 | 2.902 ± 0.043 | 1.869 ± 0.050 | 2.050 ± 0.068 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.052 ± 0.001 | 0.050 ± 0.003 | 0.031 ± 0.008 | 0.053 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.726 ± 0.025 | 0.726 ± 0.011 | 0.473 ± 0.019 | 0.546 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.183 ± 0.014 | 0.187 ± 0.013 | 0.118 ± 0.004 | 0.148 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.021 ± 0.016 | 0.984 ± 0.044 | 0.120 ± 0.002 | 0.188 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.001 | 0.022 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.255 ± 0.008 | 0.252 ± 0.008 | 0.032 ± 0.001 | 0.062 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.068 ± 0.001 | 0.069 ± 0.001 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.248 ± 0.088 | 1.201 ± 0.059 | 0.276 ± 0.009 | 0.359 ± 0.018 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.001 | 0.025 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.301 ± 0.005 | 0.305 ± 0.006 | 0.071 ± 0.002 | 0.102 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.001 | 0.081 ± 0.001 | 0.019 ± 0.001 | 0.042 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 1.672 ± 0.045 | 1.685 ± 0.034 | 0.682 ± 0.023 | 0.789 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.030 ± 0.001 | 0.033 ± 0.000 | 0.013 ± 0.000 | 0.033 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.426 ± 0.010 | 0.408 ± 0.038 | 0.175 ± 0.006 | 0.212 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.108 ± 0.001 | 0.111 ± 0.001 | 0.045 ± 0.001 | 0.071 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.066 ± 0.034 | 2.673 ± 0.036 | 0.633 ± 0.019 | 0.805 ± 0.041 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.040 ± 0.001 | 0.060 ± 0.001 | 0.016 ± 0.003 | 0.046 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.529 ± 0.007 | 0.685 ± 0.014 | 0.163 ± 0.005 | 0.220 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.141 ± 0.003 | 0.189 ± 0.004 | 0.045 ± 0.004 | 0.078 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.987 ± 0.021 | 1.965 ± 0.016 | 0.066 ± 0.005 | 0.116 ± 0.025 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.022 ± 0.000 | 0.048 ± 0.001 | 0.007 ± 0.000 | 0.006 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.249 ± 0.003 | 0.506 ± 0.008 | 0.021 ± 0.001 | 0.044 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.067 ± 0.007 | 0.144 ± 0.006 | 0.010 ± 0.000 | 0.026 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.072 ± 0.022 | 2.009 ± 0.063 | 0.104 ± 0.007 | 0.173 ± 0.025 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.049 ± 0.002 | 0.007 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.268 ± 0.005 | 0.515 ± 0.019 | 0.030 ± 0.001 | 0.052 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.072 ± 0.001 | 0.138 ± 0.011 | 0.012 ± 0.000 | 0.028 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.296 ± 0.030 | 2.165 ± 0.050 | 0.228 ± 0.007 | 0.287 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.053 ± 0.001 | 0.009 ± 0.001 | 0.026 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.332 ± 0.008 | 0.559 ± 0.015 | 0.061 ± 0.004 | 0.091 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.082 ± 0.004 | 0.152 ± 0.002 | 0.020 ± 0.001 | 0.043 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 21.670 ± 0.144 | 21.607 ± 0.141 | 20.165 ± 0.156 | 20.960 ± 0.212 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.352 ± 0.004 | 0.346 ± 0.022 | 0.328 ± 0.010 | 0.346 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.427 ± 0.094 | 5.417 ± 0.073 | 5.066 ± 0.096 | 5.224 ± 0.065 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.362 ± 0.024 | 1.359 ± 0.034 | 1.268 ± 0.047 | 1.300 ± 0.054 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.619 ± 0.011 | 1.586 ± 0.025 | 0.320 ± 0.013 | 1.070 ± 0.073 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.032 ± 0.002 | 0.031 ± 0.000 | 0.008 ± 0.000 | 0.025 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.391 ± 0.020 | 0.386 ± 0.019 | 0.082 ± 0.007 | 0.283 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.106 ± 0.006 | 0.101 ± 0.002 | 0.023 ± 0.002 | 0.071 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.822 ± 0.164 | 2.663 ± 0.065 | 1.382 ± 0.048 | 2.090 ± 0.093 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.049 ± 0.001 | 0.050 ± 0.000 | 0.025 ± 0.001 | 0.038 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.658 ± 0.033 | 0.676 ± 0.010 | 0.349 ± 0.018 | 0.525 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.167 ± 0.009 | 0.174 ± 0.005 | 0.089 ± 0.005 | 0.143 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 6.737 ± 0.161 | 6.495 ± 0.083 | 5.124 ± 0.116 | 5.748 ± 0.088 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.111 ± 0.005 | 0.107 ± 0.004 | 0.084 ± 0.010 | 0.109 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.632 ± 0.025 | 1.645 ± 0.039 | 1.292 ± 0.021 | 1.406 ± 0.027 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.409 ± 0.016 | 0.403 ± 0.015 | 0.322 ± 0.009 | 0.365 ± 0.020 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.131 ± 0.038 | 1.047 ± 0.051 | 3.854 ± 0.091 | 0.902 ± 0.023 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.094 ± 0.008 | 0.044 ± 0.006 | 0.095 ± 0.010 | 0.040 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.342 ± 0.012 | 0.266 ± 0.005 | 1.014 ± 0.027 | 0.271 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.149 ± 0.003 | 0.098 ± 0.003 | 0.274 ± 0.023 | 0.081 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.176 ± 0.003 | 0.125 ± 0.002 | 0.042 ± 0.005 | 0.030 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.070 ± 0.003 | 0.023 ± 0.000 | 0.030 ± 0.002 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.100 ± 0.012 | 0.045 ± 0.002 | 0.033 ± 0.003 | 0.021 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.072 ± 0.004 | 0.028 ± 0.000 | 0.032 ± 0.004 | 0.021 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.231 ± 0.008 | 0.165 ± 0.006 | 0.086 ± 0.016 | 0.217 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.067 ± 0.003 | 0.024 ± 0.000 | 0.029 ± 0.004 | 0.022 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.109 ± 0.002 | 0.060 ± 0.000 | 0.042 ± 0.008 | 0.068 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.003 | 0.031 ± 0.002 | 0.032 ± 0.001 | 0.032 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.601 ± 0.056 | 0.544 ± 0.031 | 3.313 ± 0.098 | 0.268 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.088 ± 0.002 | 0.036 ± 0.000 | 0.081 ± 0.003 | 0.025 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.212 ± 0.006 | 0.157 ± 0.001 | 0.839 ± 0.033 | 0.084 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.113 ± 0.002 | 0.066 ± 0.003 | 0.225 ± 0.011 | 0.040 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 8.883 ± 0.080 | 5.036 ± 0.119 | 0.951 ± 0.038 | 0.913 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.248 ± 0.008 | 0.129 ± 0.007 | 0.037 ± 0.006 | 0.040 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.289 ± 0.040 | 1.293 ± 0.032 | 0.273 ± 0.026 | 0.263 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.654 ± 0.025 | 0.365 ± 0.023 | 0.081 ± 0.009 | 0.089 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.528 ± 0.298 | 3.630 ± 0.088 | 0.118 ± 0.009 | 0.151 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.162 ± 0.002 | 0.108 ± 0.001 | 0.025 ± 0.003 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.107 ± 0.027 | 0.953 ± 0.021 | 0.046 ± 0.004 | 0.056 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.354 ± 0.007 | 0.267 ± 0.015 | 0.030 ± 0.013 | 0.030 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.669 ± 0.213 | 3.726 ± 0.095 | 0.172 ± 0.015 | 0.203 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.162 ± 0.004 | 0.103 ± 0.001 | 0.025 ± 0.001 | 0.026 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.193 ± 0.032 | 0.983 ± 0.025 | 0.060 ± 0.004 | 0.064 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.366 ± 0.013 | 0.283 ± 0.002 | 0.032 ± 0.001 | 0.033 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.426 ± 0.034 | 3.985 ± 0.112 | 0.391 ± 0.107 | 0.338 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.192 ± 0.004 | 0.107 ± 0.001 | 0.029 ± 0.002 | 0.028 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.425 ± 0.016 | 1.048 ± 0.040 | 0.105 ± 0.020 | 0.111 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.432 ± 0.017 | 0.289 ± 0.015 | 0.043 ± 0.002 | 0.045 ± 0.012 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.628 ± 0.017 | 0.617 ± 0.026 | 0.633 ± 0.016 | 0.714 ± 0.019 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.708 ± 0.018 | 2.645 ± 0.050 | 2.720 ± 0.035 | 2.895 ± 0.056 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.202 ± 0.003 | 0.205 ± 0.006 | 0.207 ± 0.018 | 0.232 ± 0.024 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 74.347 ± 0.350 | 82.103 ± 1.133 | 136.261 ± 9.167 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 2.924 ± 0.043 | 3.098 ± 0.064 | 3.939 ± 0.042 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 13.582 ± 0.056 | 14.580 ± 0.124 | 21.513 ± 0.071 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 74.757 ± 0.219 | 70.857 ± 1.101 | 134.355 ± 0.581 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 2.879 ± 0.048 | 2.847 ± 0.062 | 3.922 ± 0.067 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 13.559 ± 0.099 | 13.091 ± 0.151 | 21.220 ± 0.100 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 28.958 ± 1.979 | 61.912 ± 1.680 | 86.958 ± 0.430 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.712 ± 0.057 | 1.315 ± 0.063 | 1.744 ± 0.047 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 4.235 ± 0.196 | 8.494 ± 0.078 | 11.950 ± 0.052 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 29.216 ± 0.327 | 32.476 ± 0.914 | 88.168 ± 0.530 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.714 ± 0.034 | 0.789 ± 0.038 | 1.758 ± 0.028 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 4.261 ± 0.065 | 4.676 ± 0.121 | 11.852 ± 0.075 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.031 ± 0.001 | 0.033 ± 0.001 | 0.020 ± 0.003 | 0.101 ± 0.009 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.014 ± 0.002 | 0.013 ± 0.001 | 0.006 ± 0.001 | 0.033 ± 0.006 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.143 ± 0.004 | 0.096 ± 0.001 | 0.067 ± 0.015 | 0.248 ± 0.013 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.096 ± 0.003 | 0.040 ± 0.001 | 0.027 ± 0.003 | 0.055 ± 0.007 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 50.357 ± 0.944 | 56.874 ± 1.046 | 138.353 ± 3.481 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.859 ± 0.046 | 1.948 ± 0.085 | 3.181 ± 0.037 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 8.865 ± 0.057 | 10.020 ± 0.112 | 19.141 ± 0.171 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 50.690 ± 0.267 | 56.327 ± 1.825 | 138.443 ± 0.422 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.853 ± 0.039 | 1.856 ± 0.079 | 3.158 ± 0.046 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 8.855 ± 0.037 | 9.462 ± 0.068 | 19.586 ± 0.150 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.414 ± 0.006 | 0.138 ± 0.002 | 0.084 ± 0.014 | 0.083 ± 0.011 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.192 ± 0.007 | 0.076 ± 0.001 | 0.040 ± 0.004 | 0.042 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 5.769 ± 0.145 | 6.054 ± 0.382 | 5.562 ± 0.235 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.297 ± 0.012 | 0.299 ± 0.028 | 0.244 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.237 ± 0.054 | 1.207 ± 0.105 | 1.018 ± 0.022 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 5.757 ± 0.150 | 9.719 ± 0.135 | 4.887 ± 0.300 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.287 ± 0.010 | 0.389 ± 0.013 | 0.238 ± 0.009 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.192 ± 0.042 | 1.947 ± 0.059 | 0.976 ± 0.014 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 2.578 ± 0.051 | 1.263 ± 0.034 | 1.265 ± 0.038 | 1.435 ± 0.023 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.625 ± 0.011 | 0.306 ± 0.012 | 0.299 ± 0.014 | 0.337 ± 0.032 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 117.975 ± 0.321 | 137.658 ± 2.976 | 179.347 ± 0.818 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.036 ± 0.024 | 5.530 ± 0.031 | 6.105 ± 0.043 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 23.213 ± 0.101 | 26.135 ± 0.188 | 31.155 ± 0.101 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 117.264 ± 0.571 | 114.222 ± 1.237 | 179.820 ± 3.678 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 5.030 ± 0.050 | 5.056 ± 0.070 | 6.054 ± 0.049 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 23.153 ± 0.114 | 22.765 ± 0.091 | 30.993 ± 0.110 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 5.269 ± 0.782 | 5.112 ± 0.226 | 5.184 ± 1.094 | 33.250 ± 0.192 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.039 ± 0.006 | 0.026 ± 0.000 | 0.016 ± 0.003 | 0.103 ± 0.020 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.191 ± 0.002 | 0.105 ± 0.002 | 0.091 ± 0.009 | 0.562 ± 0.021 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.992 ± 0.150 | 0.656 ± 0.025 | 0.652 ± 0.035 | 4.182 ± 0.076 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.311 ± 0.031 | 1.315 ± 0.024 | 1.291 ± 0.026 | 8.203 ± 0.058 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.369 ± 0.026 | 0.379 ± 0.015 | 0.366 ± 0.016 | 2.166 ± 0.036 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.377 ± 0.009 | 0.377 ± 0.007 | 0.403 ± 0.009 | 0.414 ± 0.027 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.162 ± 0.025 | 1.171 ± 0.019 | 1.359 ± 0.019 | 1.296 ± 0.022 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.070 ± 0.002 | 0.072 ± 0.002 | 0.068 ± 0.002 | 0.090 ± 0.012 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.073 ± 0.006 | 0.071 ± 0.001 | 0.054 ± 0.012 | 0.078 ± 0.014 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.081 ± 0.004 | 0.079 ± 0.002 | 0.052 ± 0.005 | 0.081 ± 0.016 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.103 ± 0.002 | 0.095 ± 0.001 | 0.063 ± 0.005 | 0.103 ± 0.019 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.300 ± 0.006 | 0.214 ± 0.006 | 0.204 ± 0.013 | 0.246 ± 0.024 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.318 ± 0.029 | 0.216 ± 0.002 | 0.203 ± 0.010 | 0.260 ± 0.018 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.383 ± 0.025 | 0.269 ± 0.007 | 0.244 ± 0.018 | 0.297 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.025 ± 0.001 | 0.035 ± 0.001 | 0.018 ± 0.001 | 0.038 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.029 ± 0.004 | 0.037 ± 0.002 | 0.019 ± 0.000 | 0.047 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.037 ± 0.000 | 0.044 ± 0.001 | 0.026 ± 0.004 | 0.049 ± 0.009 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.215 ± 0.023 | 1.228 ± 0.023 | 1.238 ± 0.016 | 1.286 ± 0.033 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.847 ± 0.043 | 4.844 ± 0.035 | 4.877 ± 0.049 | 4.959 ± 0.065 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.279 ± 0.023 | 0.281 ± 0.004 | 0.281 ± 0.009 | 0.345 ± 0.049 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 1.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 29.8x faster than the slowest successful cell (`tenferro-trace`, 1.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 24.4x faster than the slowest successful cell (`tenferro-trace`, 0.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 2.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 2.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 17.3x faster than the slowest successful cell (`tenferro-trace`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 3.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 30.0x faster than the slowest successful cell (`tenferro-eager`, 4.528 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 38.3x faster than the slowest successful cell (`tenferro-eager`, 4.528 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 1.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 24.2x faster than the slowest successful cell (`tenferro-eager`, 1.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.1x faster than the slowest successful cell (`tenferro-eager`, 4.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 27.1x faster than the slowest successful cell (`tenferro-eager`, 4.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 1.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`tenferro-eager`, 1.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 5.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 5.426 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 1.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 1.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_103859/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260916_103859`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260916_103859`.

- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

### CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_103859/cpu_ops_t4_20260916_103859.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_103859/cpu_ops_t4_20260916_103859.md`

### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 11.143 ± 0.046 | 11.197 ± 0.062 | 10.296 ± 0.177 | 5.320 ± 0.035 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.186 ± 0.003 | 0.214 ± 0.017 | 0.162 ± 0.001 | 0.240 ± 0.212 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.789 ± 0.044 | 2.800 ± 0.025 | 2.573 ± 0.036 | 1.102 ± 0.049 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.698 ± 0.006 | 0.737 ± 0.017 | 0.652 ± 0.023 | 0.484 ± 0.146 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.967 ± 0.013 | 1.037 ± 0.043 | 0.157 ± 0.009 | 0.295 ± 0.022 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.003 | 0.050 ± 0.006 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.252 ± 0.005 | 0.273 ± 0.004 | 0.051 ± 0.025 | 0.092 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.075 ± 0.002 | 0.095 ± 0.009 | 0.011 ± 0.000 | 0.039 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.556 ± 0.016 | 1.643 ± 0.066 | 0.720 ± 0.024 | 0.610 ± 0.162 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.003 | 0.062 ± 0.007 | 0.014 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.398 ± 0.007 | 0.427 ± 0.006 | 0.198 ± 0.026 | 0.243 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.109 ± 0.003 | 0.133 ± 0.014 | 0.046 ± 0.001 | 0.073 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.519 ± 0.036 | 3.667 ± 0.057 | 2.772 ± 0.176 | 0.834 ± 0.664 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.068 ± 0.001 | 0.090 ± 0.004 | 0.043 ± 0.000 | 0.067 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.890 ± 0.007 | 0.919 ± 0.012 | 0.678 ± 0.015 | 0.549 ± 0.030 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.232 ± 0.002 | 0.254 ± 0.005 | 0.171 ± 0.007 | 0.198 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.318 ± 0.022 | 0.355 ± 0.036 | 1.055 ± 0.092 | 0.355 ± 0.053 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.019 ± 0.003 | 0.045 ± 0.004 | 0.020 ± 0.002 | 0.075 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.085 ± 0.006 | 0.120 ± 0.013 | 0.279 ± 0.012 | 0.257 ± 0.043 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.032 ± 0.003 | 0.061 ± 0.005 | 0.079 ± 0.012 | 0.114 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.047 ± 0.003 | 0.071 ± 0.004 | 0.010 ± 0.003 | 0.078 ± 0.025 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.004 | 0.048 ± 0.025 | 0.005 ± 0.000 | 0.051 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.019 ± 0.003 | 0.046 ± 0.005 | 0.006 ± 0.001 | 0.064 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.002 | 0.043 ± 0.014 | 0.005 ± 0.000 | 0.060 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.058 ± 0.003 | 0.091 ± 0.006 | 0.053 ± 0.016 | 0.212 ± 0.029 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.004 | 0.038 ± 0.003 | 0.005 ± 0.000 | 0.069 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.023 ± 0.004 | 0.050 ± 0.008 | 0.011 ± 0.006 | 0.098 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.040 ± 0.002 | 0.006 ± 0.000 | 0.077 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.120 ± 0.005 | 0.160 ± 0.015 | 0.551 ± 0.085 | 0.237 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.013 ± 0.004 | 0.043 ± 0.008 | 0.013 ± 0.000 | 0.065 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.039 ± 0.003 | 0.070 ± 0.004 | 0.153 ± 0.014 | 0.104 ± 0.042 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.019 ± 0.002 | 0.045 ± 0.004 | 0.035 ± 0.001 | 0.083 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.979 ± 0.096 | 2.976 ± 0.054 | 1.817 ± 0.025 | 1.092 ± 0.018 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.058 ± 0.002 | 0.080 ± 0.005 | 0.048 ± 0.005 | 0.057 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.743 ± 0.005 | 0.755 ± 0.010 | 0.501 ± 0.029 | 0.580 ± 0.140 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.195 ± 0.004 | 0.224 ± 0.010 | 0.139 ± 0.007 | 0.342 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.018 ± 0.010 | 1.080 ± 0.166 | 0.143 ± 0.012 | 0.184 ± 0.014 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.031 ± 0.003 | 0.050 ± 0.005 | 0.017 ± 0.003 | 0.008 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.266 ± 0.007 | 0.288 ± 0.008 | 0.052 ± 0.018 | 0.064 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.077 ± 0.004 | 0.099 ± 0.006 | 0.024 ± 0.004 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.215 ± 0.017 | 1.259 ± 0.016 | 0.307 ± 0.013 | 0.361 ± 0.021 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.034 ± 0.002 | 0.053 ± 0.005 | 0.021 ± 0.003 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.311 ± 0.003 | 0.337 ± 0.015 | 0.095 ± 0.013 | 0.111 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.111 ± 0.008 | 0.034 ± 0.002 | 0.043 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 1.687 ± 0.021 | 1.743 ± 0.053 | 0.713 ± 0.008 | 0.417 ± 0.040 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.039 ± 0.001 | 0.060 ± 0.005 | 0.028 ± 0.001 | 0.031 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.434 ± 0.003 | 0.452 ± 0.008 | 0.195 ± 0.011 | 0.218 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.119 ± 0.002 | 0.140 ± 0.005 | 0.065 ± 0.007 | 0.071 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.053 ± 0.013 | 2.734 ± 0.036 | 0.325 ± 0.013 | 0.570 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.049 ± 0.004 | 0.094 ± 0.007 | 0.031 ± 0.005 | 0.040 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.533 ± 0.005 | 0.716 ± 0.021 | 0.129 ± 0.013 | 0.280 ± 0.030 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.146 ± 0.003 | 0.223 ± 0.009 | 0.050 ± 0.008 | 0.167 ± 0.025 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.994 ± 0.007 | 2.179 ± 0.381 | 0.081 ± 0.012 | 0.107 ± 0.024 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.030 ± 0.002 | 0.083 ± 0.008 | 0.023 ± 0.004 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.261 ± 0.007 | 0.548 ± 0.018 | 0.053 ± 0.035 | 0.047 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.077 ± 0.003 | 0.179 ± 0.011 | 0.026 ± 0.003 | 0.028 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.066 ± 0.020 | 2.108 ± 0.057 | 0.113 ± 0.009 | 0.156 ± 0.015 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.031 ± 0.002 | 0.085 ± 0.013 | 0.021 ± 0.005 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.277 ± 0.003 | 0.555 ± 0.010 | 0.057 ± 0.011 | 0.061 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.082 ± 0.002 | 0.180 ± 0.008 | 0.027 ± 0.003 | 0.035 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.284 ± 0.017 | 2.279 ± 0.081 | 0.175 ± 0.037 | 0.210 ± 0.027 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.035 ± 0.004 | 0.084 ± 0.004 | 0.026 ± 0.002 | 0.025 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.335 ± 0.005 | 0.602 ± 0.008 | 0.076 ± 0.015 | 0.096 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.095 ± 0.003 | 0.189 ± 0.006 | 0.036 ± 0.014 | 0.041 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 21.915 ± 0.831 | 22.280 ± 0.364 | 20.602 ± 0.304 | 10.716 ± 0.068 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.356 ± 0.003 | 0.395 ± 0.012 | 0.321 ± 0.010 | 0.298 ± 0.112 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.518 ± 0.083 | 5.568 ± 0.035 | 5.189 ± 0.109 | 1.972 ± 0.904 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.369 ± 0.018 | 1.440 ± 0.027 | 1.319 ± 0.042 | 0.683 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.638 ± 0.028 | 1.667 ± 0.033 | 0.346 ± 0.018 | 1.099 ± 0.035 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.042 ± 0.003 | 0.072 ± 0.004 | 0.009 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.418 ± 0.005 | 0.455 ± 0.011 | 0.088 ± 0.012 | 0.270 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.116 ± 0.002 | 0.144 ± 0.006 | 0.024 ± 0.000 | 0.072 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.718 ± 0.013 | 2.845 ± 0.063 | 1.442 ± 0.054 | 1.049 ± 0.052 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.057 ± 0.003 | 0.094 ± 0.009 | 0.025 ± 0.000 | 0.040 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.692 ± 0.008 | 0.727 ± 0.024 | 0.360 ± 0.029 | 0.544 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.181 ± 0.004 | 0.218 ± 0.011 | 0.091 ± 0.001 | 0.151 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.575 ± 0.026 | 6.813 ± 0.085 | 5.546 ± 0.202 | 2.111 ± 0.206 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.119 ± 0.003 | 0.149 ± 0.008 | 0.084 ± 0.001 | 0.112 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.660 ± 0.014 | 1.687 ± 0.012 | 1.347 ± 0.072 | 0.736 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.423 ± 0.003 | 0.466 ± 0.017 | 0.335 ± 0.017 | 0.477 ± 0.153 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.190 ± 0.020 | 1.034 ± 0.044 | 3.970 ± 0.170 | 0.624 ± 0.036 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.243 ± 0.006 | 0.166 ± 0.020 | 0.093 ± 0.005 | 0.042 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.514 ± 0.014 | 0.403 ± 0.011 | 1.130 ± 0.072 | 0.447 ± 0.018 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.298 ± 0.014 | 0.227 ± 0.014 | 0.310 ± 0.029 | 0.085 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.337 ± 0.014 | 0.309 ± 0.056 | 0.046 ± 0.021 | 0.030 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.216 ± 0.013 | 0.140 ± 0.007 | 0.030 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.246 ± 0.024 | 0.167 ± 0.012 | 0.036 ± 0.040 | 0.021 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.224 ± 0.013 | 0.148 ± 0.006 | 0.031 ± 0.001 | 0.019 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.382 ± 0.015 | 0.318 ± 0.013 | 0.173 ± 0.055 | 0.291 ± 0.019 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.213 ± 0.007 | 0.143 ± 0.018 | 0.031 ± 0.005 | 0.024 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.253 ± 0.005 | 0.189 ± 0.010 | 0.067 ± 0.031 | 0.089 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.230 ± 0.012 | 0.150 ± 0.049 | 0.033 ± 0.002 | 0.042 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.759 ± 0.014 | 0.772 ± 0.132 | 3.951 ± 0.580 | 0.255 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.228 ± 0.007 | 0.155 ± 0.018 | 0.082 ± 0.001 | 0.027 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.369 ± 0.003 | 0.292 ± 0.009 | 0.892 ± 0.065 | 0.088 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.250 ± 0.010 | 0.181 ± 0.036 | 0.257 ± 0.031 | 0.042 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 8.877 ± 0.044 | 5.264 ± 0.172 | 0.598 ± 0.024 | 0.717 ± 0.030 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.436 ± 0.006 | 0.251 ± 0.012 | 0.060 ± 0.014 | 0.041 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.472 ± 0.017 | 1.434 ± 0.023 | 0.314 ± 0.027 | 0.330 ± 0.029 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.854 ± 0.013 | 0.489 ± 0.009 | 0.115 ± 0.025 | 0.186 ± 0.025 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.281 ± 0.027 | 3.890 ± 0.038 | 0.192 ± 0.021 | 0.137 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.357 ± 0.012 | 0.234 ± 0.043 | 0.045 ± 0.009 | 0.010 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.298 ± 0.010 | 1.089 ± 0.026 | 0.097 ± 0.027 | 0.054 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.550 ± 0.010 | 0.403 ± 0.020 | 0.049 ± 0.005 | 0.028 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.584 ± 0.038 | 4.076 ± 0.095 | 0.251 ± 0.046 | 0.202 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.364 ± 0.018 | 0.231 ± 0.032 | 0.045 ± 0.004 | 0.026 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.374 ± 0.022 | 1.110 ± 0.015 | 0.120 ± 0.019 | 0.071 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.572 ± 0.012 | 0.408 ± 0.011 | 0.052 ± 0.011 | 0.036 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.538 ± 0.039 | 4.357 ± 0.232 | 0.465 ± 0.061 | 0.303 ± 0.052 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.375 ± 0.029 | 0.241 ± 0.021 | 0.048 ± 0.006 | 0.027 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.625 ± 0.011 | 1.184 ± 0.019 | 0.129 ± 0.019 | 0.109 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.632 ± 0.013 | 0.440 ± 0.024 | 0.075 ± 0.018 | 0.048 ± 0.009 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.640 ± 0.017 | 0.654 ± 0.017 | 0.615 ± 0.033 | 0.699 ± 0.031 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.696 ± 0.022 | 2.673 ± 0.037 | 2.703 ± 0.036 | 2.843 ± 0.026 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.221 ± 0.015 | 0.236 ± 0.010 | 0.201 ± 0.019 | 0.230 ± 0.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 76.286 ± 8.273 | 80.619 ± 0.689 | 85.989 ± 0.693 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 3.021 ± 0.026 | 3.372 ± 0.241 | 3.122 ± 0.062 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 14.266 ± 1.104 | 14.836 ± 0.087 | 15.370 ± 0.100 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 74.391 ± 4.684 | 71.138 ± 4.576 | 86.042 ± 0.278 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 2.990 ± 0.018 | 2.988 ± 0.375 | 3.088 ± 0.082 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 13.410 ± 0.111 | 13.362 ± 0.216 | 15.367 ± 0.099 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 31.104 ± 0.957 | 67.921 ± 20.734 | 36.060 ± 0.345 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.912 ± 0.064 | 1.389 ± 0.126 | 0.793 ± 0.041 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.386 ± 0.221 | 8.237 ± 0.426 | 5.477 ± 0.200 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 30.232 ± 0.488 | 32.958 ± 1.423 | 36.520 ± 0.550 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.866 ± 0.019 | 0.867 ± 0.081 | 0.837 ± 0.060 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.738 ± 2.670 | 4.556 ± 0.092 | 5.775 ± 0.486 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.050 ± 0.005 | 0.078 ± 0.006 | 0.019 ± 0.003 | 0.110 ± 0.005 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.036 ± 0.003 | 0.052 ± 0.003 | 0.006 ± 0.001 | 0.036 ± 0.008 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.296 ± 0.039 | 0.200 ± 0.013 | 0.066 ± 0.015 | 0.219 ± 0.029 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.246 ± 0.019 | 0.150 ± 0.004 | 0.026 ± 0.004 | 0.059 ± 0.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 49.491 ± 0.993 | 58.586 ± 16.150 | 60.426 ± 0.452 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 2.035 ± 0.044 | 1.871 ± 0.042 | 1.807 ± 0.112 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 8.735 ± 0.144 | 8.923 ± 0.106 | 10.191 ± 0.357 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 52.081 ± 3.514 | 61.045 ± 17.880 | 64.384 ± 0.291 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 1.986 ± 0.015 | 1.804 ± 0.073 | 1.929 ± 0.031 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 8.349 ± 0.099 | 8.759 ± 0.213 | 10.427 ± 0.120 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.596 ± 0.025 | 0.234 ± 0.004 | 0.099 ± 0.043 | 0.091 ± 0.007 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.388 ± 0.016 | 0.178 ± 0.009 | 0.043 ± 0.005 | 0.044 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 6.144 ± 0.387 | 5.787 ± 0.178 | 5.348 ± 0.371 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.409 ± 0.028 | 0.294 ± 0.021 | 0.237 ± 0.018 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.700 ± 0.164 | 1.120 ± 0.123 | 1.250 ± 0.129 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.972 ± 0.184 | 9.586 ± 0.388 | 5.032 ± 0.330 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.372 ± 0.012 | 0.406 ± 0.050 | 0.231 ± 0.014 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.396 ± 0.086 | 1.781 ± 0.090 | 1.087 ± 0.122 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 2.613 ± 0.018 | 1.339 ± 0.019 | 1.281 ± 0.029 | 1.449 ± 0.035 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.731 ± 0.021 | 0.378 ± 0.011 | 0.302 ± 0.018 | 0.333 ± 0.037 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 123.011 ± 10.576 | 133.014 ± 2.689 | 129.844 ± 1.382 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.123 ± 0.032 | 5.922 ± 0.516 | 5.384 ± 0.051 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 23.233 ± 0.188 | 26.387 ± 0.581 | 24.728 ± 0.175 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 118.477 ± 2.470 | 113.138 ± 0.576 | 129.953 ± 0.633 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 5.083 ± 0.023 | 5.105 ± 0.041 | 5.330 ± 0.052 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 23.149 ± 1.383 | 23.462 ± 0.838 | 24.532 ± 0.205 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.893 ± 0.160 | 4.758 ± 0.223 | 5.582 ± 1.637 | 8.744 ± 0.097 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.064 ± 0.006 | 0.058 ± 0.003 | 0.023 ± 0.006 | 0.105 ± 0.011 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.213 ± 0.058 | 0.141 ± 0.003 | 0.111 ± 0.017 | 0.200 ± 0.017 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.904 ± 0.122 | 0.681 ± 0.021 | 0.690 ± 0.077 | 1.175 ± 0.071 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.290 ± 0.009 | 1.306 ± 0.015 | 1.299 ± 0.031 | 2.156 ± 0.045 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.383 ± 0.033 | 0.378 ± 0.009 | 0.401 ± 0.047 | 0.641 ± 0.070 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.383 ± 0.007 | 0.406 ± 0.004 | 0.413 ± 0.022 | 0.415 ± 0.018 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.160 ± 0.009 | 1.195 ± 0.011 | 1.234 ± 0.018 | 1.272 ± 0.027 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.080 ± 0.003 | 0.102 ± 0.010 | 0.088 ± 0.011 | 0.093 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.086 ± 0.005 | 0.099 ± 0.005 | 0.056 ± 0.012 | 0.084 ± 0.012 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.089 ± 0.006 | 0.100 ± 0.006 | 0.052 ± 0.009 | 0.085 ± 0.009 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.111 ± 0.002 | 0.121 ± 0.002 | 0.067 ± 0.008 | 0.106 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.298 ± 0.008 | 0.241 ± 0.013 | 0.188 ± 0.014 | 0.251 ± 0.015 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.312 ± 0.008 | 0.242 ± 0.008 | 0.191 ± 0.006 | 0.256 ± 0.025 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.354 ± 0.006 | 0.298 ± 0.007 | 0.227 ± 0.022 | 0.292 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.033 ± 0.004 | 0.061 ± 0.006 | 0.019 ± 0.005 | 0.041 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.036 ± 0.002 | 0.064 ± 0.007 | 0.021 ± 0.004 | 0.044 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.045 ± 0.002 | 0.073 ± 0.004 | 0.026 ± 0.008 | 0.053 ± 0.011 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.245 ± 0.023 | 1.272 ± 0.014 | 1.219 ± 0.031 | 1.283 ± 0.027 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.847 ± 0.021 | 4.866 ± 0.022 | 4.893 ± 0.050 | 4.953 ± 0.040 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.302 ± 0.010 | 0.333 ± 0.017 | 0.288 ± 0.038 | 0.364 ± 0.024 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_eigh` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 20.4x faster than the slowest successful cell (`tenferro-trace`, 2.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 26.8x faster than the slowest successful cell (`tenferro-trace`, 2.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 2.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 2.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 2.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 2.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 38.7x faster than the slowest successful cell (`tenferro-eager`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 3.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 8.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 8.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.436 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 31.4x faster than the slowest successful cell (`tenferro-eager`, 4.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`tenferro-eager`, 4.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 37.3x faster than the slowest successful cell (`tenferro-eager`, 0.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 1.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 1.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 4.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 4.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 1.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 1.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 5.538 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 5.538 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 0.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 1.625 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 1.625 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

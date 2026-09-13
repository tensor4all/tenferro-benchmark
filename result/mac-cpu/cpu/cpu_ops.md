# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260913_124035`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260913_124035`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

## CPU Information

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_124035/cpu_ops_t4_20260913_124035.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_124035/cpu_ops_t4_20260913_124035.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 12.464 ± 0.051 | 11.112 ± 0.058 | 49.115 ± 0.153 | 37.011 ± 1.375 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.340 ± 0.016 | 0.208 ± 0.004 | 0.772 ± 0.060 | 0.800 ± 0.118 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 3.253 ± 0.030 | 2.808 ± 0.028 | 12.039 ± 0.489 | 9.894 ± 0.207 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.914 ± 0.017 | 0.725 ± 0.010 | 3.116 ± 0.193 | 2.642 ± 0.197 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.181 ± 0.024 | 0.997 ± 0.009 | 4.307 ± 0.105 | 4.504 ± 0.116 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.153 ± 0.012 | 0.044 ± 0.003 | 0.072 ± 0.004 | 0.094 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.398 ± 0.021 | 0.277 ± 0.003 | 1.065 ± 0.024 | 1.169 ± 0.058 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.188 ± 0.011 | 0.092 ± 0.002 | 0.273 ± 0.009 | 0.330 ± 0.054 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 1.848 ± 0.036 | 1.580 ± 0.015 | 6.429 ± 0.072 | 6.612 ± 0.233 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.167 ± 0.016 | 0.059 ± 0.010 | 0.108 ± 0.009 | 0.129 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.555 ± 0.015 | 0.421 ± 0.009 | 1.655 ± 0.085 | 1.737 ± 0.074 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.228 ± 0.014 | 0.131 ± 0.002 | 0.408 ± 0.017 | 0.473 ± 0.046 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 4.019 ± 0.045 | 3.551 ± 0.029 | 14.124 ± 0.237 | 11.899 ± 0.236 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.231 ± 0.037 | 0.087 ± 0.005 | 0.224 ± 0.005 | 0.297 ± 0.035 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 1.112 ± 0.016 | 0.909 ± 0.026 | 3.665 ± 0.200 | 3.400 ± 0.244 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.368 ± 0.008 | 0.251 ± 0.003 | 0.893 ± 0.041 | 0.954 ± 0.083 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 0.624 ± 0.036 | 0.326 ± 0.023 | 53.979 ± 1.012 | 53.277 ± 1.223 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.170 ± 0.024 | 0.048 ± 0.002 | 0.881 ± 0.083 | 1.017 ± 0.076 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.278 ± 0.014 | 0.124 ± 0.013 | 14.771 ± 0.625 | 13.667 ± 0.365 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.161 ± 0.009 | 0.062 ± 0.005 | 3.475 ± 0.318 | 3.578 ± 0.138 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 0.191 ± 0.014 | 0.079 ± 0.003 | 0.874 ± 0.017 | 0.973 ± 0.095 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.151 ± 0.013 | 0.048 ± 0.003 | 0.025 ± 0.002 | 0.111 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.162 ± 0.014 | 0.051 ± 0.011 | 0.226 ± 0.001 | 0.334 ± 0.037 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.158 ± 0.021 | 0.046 ± 0.002 | 0.065 ± 0.001 | 0.164 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 0.215 ± 0.028 | 0.091 ± 0.005 | 3.341 ± 0.042 | 3.644 ± 0.108 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.147 ± 0.005 | 0.045 ± 0.002 | 0.067 ± 0.003 | 0.162 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.166 ± 0.024 | 0.054 ± 0.001 | 0.868 ± 0.044 | 0.986 ± 0.117 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.145 ± 0.007 | 0.047 ± 0.002 | 0.236 ± 0.022 | 0.373 ± 0.061 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 0.307 ± 0.015 | 0.153 ± 0.004 | 13.746 ± 0.158 | 13.559 ± 0.294 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.145 ± 0.009 | 0.046 ± 0.003 | 0.239 ± 0.012 | 0.351 ± 0.069 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.196 ± 0.016 | 0.073 ± 0.003 | 3.472 ± 0.104 | 3.557 ± 0.168 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.142 ± 0.014 | 0.051 ± 0.003 | 0.891 ± 0.020 | 1.014 ± 0.067 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 3.275 ± 0.022 | 2.945 ± 0.038 | 33.918 ± 0.234 | 32.795 ± 2.374 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.196 ± 0.011 | 0.078 ± 0.002 | 0.562 ± 0.033 | 0.598 ± 0.083 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.934 ± 0.022 | 0.766 ± 0.006 | 9.108 ± 0.173 | 8.475 ± 0.187 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.346 ± 0.023 | 0.219 ± 0.007 | 2.195 ± 0.232 | 2.214 ± 0.079 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.206 ± 0.022 | 1.054 ± 0.012 | 2.882 ± 0.078 | 2.939 ± 0.128 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.149 ± 0.009 | 0.054 ± 0.011 | 0.071 ± 0.028 | 0.070 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.406 ± 0.014 | 0.290 ± 0.005 | 0.716 ± 0.018 | 0.768 ± 0.071 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.205 ± 0.017 | 0.100 ± 0.004 | 0.192 ± 0.014 | 0.202 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 1.424 ± 0.038 | 1.240 ± 0.021 | 4.451 ± 0.115 | 4.657 ± 0.162 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.153 ± 0.014 | 0.052 ± 0.003 | 0.094 ± 0.007 | 0.095 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.466 ± 0.024 | 0.335 ± 0.007 | 1.187 ± 0.096 | 1.201 ± 0.050 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.206 ± 0.016 | 0.110 ± 0.005 | 0.294 ± 0.025 | 0.335 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 1.921 ± 0.044 | 1.698 ± 0.025 | 10.541 ± 0.541 | 10.429 ± 0.256 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.166 ± 0.009 | 0.063 ± 0.005 | 0.189 ± 0.018 | 0.212 ± 0.023 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.603 ± 0.007 | 0.452 ± 0.015 | 2.665 ± 0.077 | 2.763 ± 0.154 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.238 ± 0.009 | 0.137 ± 0.002 | 0.670 ± 0.021 | 0.725 ± 0.038 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | 3.471 ± 0.026 | 2.758 ± 0.045 | 41.076 ± 0.607 | 38.691 ± 1.563 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | 0.205 ± 0.010 | 0.125 ± 0.002 | 0.678 ± 0.026 | 0.720 ± 0.086 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | 0.993 ± 0.015 | 0.764 ± 0.011 | 10.623 ± 0.224 | 9.909 ± 0.428 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | 0.366 ± 0.015 | 0.264 ± 0.009 | 2.566 ± 0.157 | 2.628 ± 0.163 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | 1.197 ± 0.018 | 2.047 ± 0.034 | 4.368 ± 0.097 | 4.601 ± 0.253 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.163 ± 0.011 | 0.119 ± 0.005 | 0.094 ± 0.007 | 0.111 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | 0.416 ± 0.008 | 0.580 ± 0.013 | 1.137 ± 0.041 | 1.188 ± 0.135 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.201 ± 0.015 | 0.207 ± 0.006 | 0.303 ± 0.018 | 0.369 ± 0.093 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | 1.331 ± 0.035 | 2.097 ± 0.021 | 6.160 ± 0.071 | 6.406 ± 0.142 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.164 ± 0.019 | 0.115 ± 0.004 | 0.133 ± 0.005 | 0.140 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | 0.442 ± 0.012 | 0.594 ± 0.014 | 1.633 ± 0.048 | 1.647 ± 0.127 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.218 ± 0.013 | 0.217 ± 0.011 | 0.421 ± 0.022 | 0.463 ± 0.042 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | 1.747 ± 0.019 | 2.259 ± 0.038 | 12.442 ± 0.142 | 12.570 ± 0.477 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | 0.162 ± 0.011 | 0.119 ± 0.004 | 0.231 ± 0.031 | 0.261 ± 0.029 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | 0.553 ± 0.015 | 0.634 ± 0.013 | 3.241 ± 0.374 | 3.260 ± 0.099 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | 0.235 ± 0.007 | 0.225 ± 0.006 | 0.797 ± 0.026 | 0.918 ± 0.050 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 22.157 ± 0.039 | 21.938 ± 0.073 | 52.608 ± 0.130 | 33.895 ± 1.685 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.491 ± 0.059 | 0.379 ± 0.004 | 0.842 ± 0.074 | 0.866 ± 0.072 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 5.652 ± 0.022 | 5.513 ± 0.021 | 14.298 ± 0.368 | 8.706 ± 0.266 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 1.528 ± 0.018 | 1.403 ± 0.017 | 3.337 ± 0.177 | 2.385 ± 0.116 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.779 ± 0.019 | 1.625 ± 0.024 | 3.117 ± 0.179 | 3.934 ± 0.161 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.168 ± 0.019 | 0.065 ± 0.006 | 0.055 ± 0.003 | 0.087 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.559 ± 0.014 | 0.435 ± 0.008 | 0.772 ± 0.022 | 1.020 ± 0.086 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.255 ± 0.027 | 0.140 ± 0.002 | 0.196 ± 0.007 | 0.259 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 2.922 ± 0.041 | 2.701 ± 0.052 | 5.680 ± 0.079 | 5.425 ± 0.261 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.190 ± 0.033 | 0.083 ± 0.006 | 0.097 ± 0.003 | 0.128 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.839 ± 0.022 | 0.707 ± 0.021 | 1.465 ± 0.054 | 1.703 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.318 ± 0.025 | 0.204 ± 0.007 | 0.361 ± 0.010 | 0.456 ± 0.028 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 6.841 ± 0.041 | 6.591 ± 0.038 | 15.080 ± 0.088 | 10.516 ± 0.353 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.249 ± 0.009 | 0.145 ± 0.004 | 0.244 ± 0.014 | 0.308 ± 0.078 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 1.824 ± 0.014 | 1.681 ± 0.010 | 3.814 ± 0.052 | 3.055 ± 0.179 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.562 ± 0.015 | 0.451 ± 0.010 | 0.957 ± 0.029 | 1.055 ± 0.068 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 1.510 ± 0.031 | 0.700 ± 0.050 | 57.333 ± 0.589 | 52.972 ± 1.174 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.394 ± 0.015 | 0.118 ± 0.008 | 0.951 ± 0.041 | 1.353 ± 0.078 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.705 ± 0.015 | 0.290 ± 0.006 | 14.814 ± 2.781 | 14.193 ± 0.333 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.453 ± 0.011 | 0.150 ± 0.004 | 3.699 ± 0.232 | 3.981 ± 0.290 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 0.489 ± 0.053 | 0.177 ± 0.009 | 0.901 ± 0.027 | 1.352 ± 0.160 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.369 ± 0.023 | 0.115 ± 0.005 | 0.052 ± 0.002 | 0.479 ± 0.043 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.396 ± 0.010 | 0.125 ± 0.011 | 0.258 ± 0.012 | 0.682 ± 0.034 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.352 ± 0.026 | 0.114 ± 0.020 | 0.092 ± 0.004 | 0.522 ± 0.039 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 0.549 ± 0.031 | 0.212 ± 0.004 | 3.434 ± 0.037 | 4.074 ± 0.239 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.364 ± 0.020 | 0.107 ± 0.004 | 0.096 ± 0.011 | 0.532 ± 0.034 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.412 ± 0.037 | 0.130 ± 0.007 | 0.992 ± 0.115 | 1.452 ± 0.135 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.373 ± 0.016 | 0.115 ± 0.009 | 0.259 ± 0.009 | 0.718 ± 0.065 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 0.995 ± 0.051 | 0.527 ± 0.033 | 16.562 ± 0.111 | 14.149 ± 0.391 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.367 ± 0.018 | 0.115 ± 0.004 | 0.307 ± 0.008 | 0.703 ± 0.055 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.527 ± 0.027 | 0.224 ± 0.013 | 4.438 ± 0.411 | 4.106 ± 0.215 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.374 ± 0.019 | 0.133 ± 0.024 | 1.109 ± 0.062 | 1.389 ± 0.203 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | 10.645 ± 0.408 | 5.178 ± 0.083 | 41.599 ± 0.682 | 39.698 ± 1.325 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | 0.625 ± 0.027 | 0.326 ± 0.010 | 0.721 ± 0.041 | 1.056 ± 0.068 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | 2.993 ± 0.057 | 1.528 ± 0.045 | 10.683 ± 0.288 | 10.655 ± 0.495 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | 1.121 ± 0.013 | 0.575 ± 0.010 | 2.646 ± 0.175 | 2.974 ± 0.167 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | 4.644 ± 0.048 | 3.821 ± 0.048 | 4.426 ± 0.084 | 5.093 ± 0.130 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.519 ± 0.029 | 0.303 ± 0.012 | 0.128 ± 0.020 | 0.496 ± 0.100 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | 1.500 ± 0.029 | 1.157 ± 0.022 | 1.173 ± 0.045 | 1.542 ± 0.092 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.717 ± 0.023 | 0.481 ± 0.020 | 0.321 ± 0.010 | 0.733 ± 0.084 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | 4.994 ± 0.099 | 3.978 ± 0.057 | 6.277 ± 0.095 | 6.948 ± 0.142 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.556 ± 0.029 | 0.309 ± 0.012 | 0.155 ± 0.013 | 0.513 ± 0.032 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | 1.577 ± 0.020 | 1.201 ± 0.058 | 1.659 ± 0.081 | 2.071 ± 0.186 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.742 ± 0.014 | 0.495 ± 0.013 | 0.445 ± 0.026 | 0.805 ± 0.067 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | 6.048 ± 0.033 | 4.243 ± 0.036 | 12.676 ± 0.065 | 13.175 ± 0.287 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | 0.547 ± 0.017 | 0.308 ± 0.017 | 0.267 ± 0.037 | 0.601 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | 1.885 ± 0.017 | 1.271 ± 0.025 | 3.320 ± 0.485 | 3.708 ± 0.243 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | 0.818 ± 0.008 | 0.496 ± 0.011 | 0.851 ± 0.040 | 1.221 ± 0.102 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | 1.255 ± 0.017 | 0.661 ± 0.016 | 2.351 ± 0.048 | 2.470 ± 0.088 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | 8.059 ± 0.147 | 2.747 ± 0.063 | 9.223 ± 0.085 | 9.800 ± 0.440 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | 0.390 ± 0.020 | 0.237 ± 0.011 | 0.640 ± 0.013 | 0.724 ± 0.062 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | 79.280 ± 2.002 | 84.966 ± 3.913 | 75.984 ± 0.875 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.144 ± 0.065 | 3.200 ± 0.103 | 3.550 ± 0.118 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 13.649 ± 0.086 | 15.826 ± 0.941 | 14.546 ± 0.091 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | 76.930 ± 3.692 | 72.221 ± 2.275 | 74.733 ± 0.309 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.166 ± 0.054 | 2.897 ± 0.098 | 3.610 ± 0.109 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 13.624 ± 0.094 | 13.733 ± 1.050 | 14.557 ± 0.297 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | 32.616 ± 2.297 | 61.391 ± 4.324 | 21.579 ± 0.473 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.925 ± 0.048 | 1.411 ± 0.115 | 1.063 ± 0.093 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 4.474 ± 0.163 | 8.170 ± 0.104 | 3.981 ± 0.145 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | 32.081 ± 1.996 | 31.195 ± 1.988 | 23.571 ± 0.811 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.888 ± 0.129 | 0.894 ± 0.037 | 1.291 ± 0.095 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 4.242 ± 0.055 | 4.526 ± 0.068 | 4.220 ± 0.174 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | 0.177 ± 0.019 | 0.080 ± 0.004 | 3.325 ± 0.043 | 3.504 ± 0.127 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.159 ± 0.016 | 0.068 ± 0.013 | 0.862 ± 0.022 | 0.918 ± 0.069 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | 0.427 ± 0.016 | 0.148 ± 0.006 | 3.301 ± 0.054 | 3.906 ± 0.210 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.384 ± 0.018 | 0.117 ± 0.007 | 0.870 ± 0.026 | 1.244 ± 0.110 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | 51.691 ± 3.883 | 54.421 ± 1.679 | 39.112 ± 1.415 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 2.142 ± 0.089 | 1.962 ± 0.055 | 2.138 ± 0.117 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 8.495 ± 0.080 | 8.671 ± 0.049 | 7.692 ± 0.188 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | 59.124 ± 3.427 | 57.644 ± 5.196 | 44.912 ± 0.885 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.161 ± 0.053 | 1.836 ± 0.123 | 2.269 ± 0.196 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 8.353 ± 0.139 | 8.349 ± 0.120 | 8.811 ± 0.236 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | 1.232 ± 0.024 | 0.323 ± 0.009 | 1.773 ± 0.066 | 2.270 ± 0.169 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.588 ± 0.022 | 0.265 ± 0.013 | 0.500 ± 0.009 | 0.881 ± 0.061 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | 5.921 ± 0.125 | 6.496 ± 0.539 | 6.133 ± 0.302 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.458 ± 0.049 | 0.375 ± 0.021 | 0.436 ± 0.073 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.303 ± 0.017 | 1.219 ± 0.069 | 1.431 ± 0.125 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | 5.826 ± 0.197 | 10.538 ± 0.375 | 6.183 ± 0.240 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.454 ± 0.046 | 0.476 ± 0.037 | 0.464 ± 0.106 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.267 ± 0.012 | 1.856 ± 0.081 | 1.480 ± 0.207 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | 2.503 ± 0.013 | 1.344 ± 0.023 | 3.011 ± 0.086 | 3.454 ± 0.172 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.873 ± 0.011 | 0.395 ± 0.019 | 0.738 ± 0.020 | 0.943 ± 0.169 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | 122.109 ± 3.053 | 139.852 ± 6.233 | 126.327 ± 0.903 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.249 ± 0.057 | 5.728 ± 0.063 | 6.100 ± 0.200 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 23.984 ± 0.462 | 25.698 ± 0.155 | 25.496 ± 0.212 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | 121.107 ± 2.146 | 117.767 ± 9.731 | 120.394 ± 1.236 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.215 ± 0.047 | 5.087 ± 0.077 | 5.876 ± 0.105 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 23.120 ± 0.048 | 22.887 ± 0.104 | 24.450 ± 0.318 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | 5.764 ± 0.270 | 4.791 ± 0.176 | 221.597 ± 3.518 | 218.644 ± 10.092 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | 0.157 ± 0.012 | 0.062 ± 0.004 | 3.279 ± 0.041 | 3.542 ± 0.267 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | 0.284 ± 0.009 | 0.145 ± 0.002 | 13.608 ± 0.264 | 13.802 ± 0.773 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | 0.981 ± 0.026 | 0.664 ± 0.014 | 53.453 ± 0.992 | 53.556 ± 2.083 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | 1.696 ± 0.021 | 1.310 ± 0.009 | 54.136 ± 1.027 | 54.022 ± 0.555 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.653 ± 0.019 | 0.377 ± 0.005 | 53.337 ± 0.750 | 57.706 ± 4.062 | - | - |
| large | `qr` | f64 | 4 | `128x128` | 0.518 ± 0.016 | 0.406 ± 0.012 | 2.001 ± 0.033 | 2.237 ± 0.069 | - | - |
| large | `qr` | f64 | 4 | `256x256` | 1.347 ± 0.047 | 1.205 ± 0.041 | 7.881 ± 0.035 | 7.946 ± 0.279 | - | - |
| large | `qr` | f64 | 4 | `64x64` | 0.224 ± 0.015 | 0.095 ± 0.005 | 0.523 ± 0.021 | 0.560 ± 0.027 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | 0.719 ± 0.014 | 0.135 ± 0.006 | 1.711 ± 0.026 | 1.888 ± 0.112 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | 0.715 ± 0.011 | 0.138 ± 0.004 | 1.901 ± 0.050 | 2.082 ± 0.164 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | 0.754 ± 0.012 | 0.150 ± 0.007 | 2.505 ± 0.043 | 2.759 ± 0.144 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | 5.736 ± 0.028 | 0.298 ± 0.021 | 6.943 ± 0.045 | 7.134 ± 0.223 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | 5.708 ± 0.027 | 0.294 ± 0.020 | 7.358 ± 0.074 | 7.739 ± 0.229 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | 5.761 ± 0.020 | 0.343 ± 0.011 | 8.598 ± 0.042 | 9.010 ± 0.206 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.203 ± 0.011 | 0.099 ± 0.008 | 0.465 ± 0.009 | 0.537 ± 0.030 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.201 ± 0.009 | 0.099 ± 0.009 | 0.567 ± 0.016 | 0.637 ± 0.053 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.214 ± 0.009 | 0.107 ± 0.008 | 0.884 ± 0.034 | 1.001 ± 0.194 | - | - |
| large | `svd` | f64 | 4 | `128x128` | 1.357 ± 0.023 | 1.257 ± 0.014 | 2.897 ± 0.032 | 3.121 ± 0.268 | - | - |
| large | `svd` | f64 | 4 | `256x256` | 5.088 ± 0.047 | 4.968 ± 0.073 | 11.297 ± 0.065 | 11.970 ± 0.310 | - | - |
| large | `svd` | f64 | 4 | `64x64` | 0.443 ± 0.010 | 0.317 ± 0.011 | 0.726 ± 0.036 | 0.807 ± 0.074 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | 0.137 ± 0.025 | 0.041 ± 0.005 | 0.051 ± 0.001 | 0.067 ± 0.004 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | 0.208 ± 0.028 | 0.032 ± 0.004 | 0.009 ± 0.000 | 0.029 ± 0.005 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | 0.169 ± 0.019 | 0.065 ± 0.003 | 0.159 ± 0.004 | 0.234 ± 0.032 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | 0.137 ± 0.018 | 0.037 ± 0.011 | 0.010 ± 0.000 | 0.028 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | 0.130 ± 0.014 | 0.036 ± 0.004 | 0.017 ± 0.001 | 0.035 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | 0.134 ± 0.015 | 0.030 ± 0.003 | 0.069 ± 0.001 | 0.159 ± 0.015 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.258 ± 0.069 | 0.032 ± 0.004 | 0.012 ± 0.001 | 0.116 ± 0.017 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | 0.140 ± 0.022 | 0.030 ± 0.004 | 0.230 ± 0.007 | 0.322 ± 0.016 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.160 ± 0.011 | 0.033 ± 0.008 | 0.014 ± 0.000 | 0.095 ± 0.014 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.149 ± 0.034 | 0.034 ± 0.003 | 0.025 ± 0.001 | 0.118 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | 0.192 ± 0.007 | 0.051 ± 0.003 | 0.129 ± 0.009 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.188 ± 0.027 | 0.036 ± 0.003 | 0.130 ± 0.016 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | 0.222 ± 0.014 | 0.075 ± 0.002 | 0.181 ± 0.022 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.189 ± 0.019 | 0.034 ± 0.001 | 0.129 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.183 ± 0.009 | 0.043 ± 0.002 | 0.130 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | 0.181 ± 0.005 | 0.046 ± 0.004 | 0.326 ± 0.016 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.173 ± 0.011 | 0.033 ± 0.005 | 0.307 ± 0.016 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | 0.216 ± 0.010 | 0.075 ± 0.004 | 0.331 ± 0.033 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.174 ± 0.017 | 0.030 ± 0.001 | 0.315 ± 0.013 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.178 ± 0.010 | 0.039 ± 0.001 | 0.319 ± 0.012 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | 0.194 ± 0.013 | 0.139 ± 0.008 | 0.187 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.187 ± 0.042 | 0.134 ± 0.010 | 0.181 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | 0.204 ± 0.010 | 0.150 ± 0.006 | 0.203 ± 0.099 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.193 ± 0.024 | 0.133 ± 0.007 | 0.188 ± 0.012 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.184 ± 0.010 | 0.139 ± 0.019 | 0.191 ± 0.023 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | 0.191 ± 0.007 | 0.102 ± 0.006 | 0.512 ± 0.024 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.192 ± 0.014 | 0.103 ± 0.014 | 0.513 ± 0.017 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | 0.211 ± 0.005 | 0.114 ± 0.012 | 0.570 ± 0.184 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.195 ± 0.014 | 0.106 ± 0.025 | 0.499 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.195 ± 0.013 | 0.103 ± 0.004 | 0.517 ± 0.043 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | 0.367 ± 0.023 | 0.106 ± 0.008 | 0.078 ± 0.002 | 0.419 ± 0.015 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.477 ± 0.068 | 0.110 ± 0.011 | 0.020 ± 0.001 | 0.350 ± 0.017 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | 0.366 ± 0.058 | 0.105 ± 0.006 | 0.240 ± 0.012 | 0.632 ± 0.079 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.380 ± 0.021 | 0.109 ± 0.016 | 0.023 ± 0.002 | 0.363 ± 0.019 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.357 ± 0.028 | 0.107 ± 0.007 | 0.039 ± 0.003 | 0.379 ± 0.018 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | 0.350 ± 0.022 | 0.083 ± 0.005 | 0.186 ± 0.004 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.351 ± 0.039 | 0.103 ± 0.025 | 0.184 ± 0.012 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | 0.404 ± 0.030 | 0.089 ± 0.009 | 0.193 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.339 ± 0.016 | 0.073 ± 0.011 | 0.184 ± 0.029 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.350 ± 0.017 | 0.081 ± 0.006 | 0.192 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | 0.368 ± 0.010 | 0.085 ± 0.009 | 0.471 ± 0.044 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.378 ± 0.052 | 0.080 ± 0.015 | 0.466 ± 0.017 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | 0.401 ± 0.024 | 0.093 ± 0.021 | 0.525 ± 0.152 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.365 ± 0.015 | 0.075 ± 0.009 | 0.458 ± 0.021 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.364 ± 0.013 | 0.073 ± 0.013 | 0.482 ± 0.034 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | 0.459 ± 0.016 | 0.236 ± 0.010 | 0.066 ± 0.002 | 0.437 ± 0.035 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.582 ± 0.046 | 0.242 ± 0.014 | 0.031 ± 0.002 | 0.386 ± 0.011 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | 0.486 ± 0.019 | 0.238 ± 0.018 | 0.157 ± 0.004 | 0.549 ± 0.122 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.475 ± 0.037 | 0.241 ± 0.017 | 0.032 ± 0.003 | 0.389 ± 0.017 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.472 ± 0.019 | 0.236 ± 0.010 | 0.040 ± 0.003 | 0.390 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | 0.176 ± 0.007 | 0.095 ± 0.005 | 0.164 ± 0.014 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.181 ± 0.019 | 0.093 ± 0.004 | 0.154 ± 0.009 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | 0.189 ± 0.010 | 0.105 ± 0.006 | 0.162 ± 0.006 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.180 ± 0.011 | 0.095 ± 0.006 | 0.150 ± 0.011 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.180 ± 0.013 | 0.094 ± 0.003 | 0.151 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | 0.173 ± 0.011 | 0.050 ± 0.003 | 0.353 ± 0.032 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.179 ± 0.011 | 0.047 ± 0.003 | 0.347 ± 0.011 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | 0.182 ± 0.004 | 0.065 ± 0.004 | 0.352 ± 0.016 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.177 ± 0.011 | 0.046 ± 0.002 | 0.335 ± 0.008 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.176 ± 0.005 | 0.047 ± 0.003 | 0.336 ± 0.020 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | 0.359 ± 0.024 | 0.130 ± 0.004 | 0.073 ± 0.003 | 0.418 ± 0.012 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.433 ± 0.023 | 0.106 ± 0.008 | 0.021 ± 0.002 | 0.395 ± 0.027 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | 0.451 ± 0.028 | 0.170 ± 0.005 | 0.200 ± 0.002 | 0.513 ± 0.042 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.324 ± 0.026 | 0.115 ± 0.013 | 0.022 ± 0.001 | 0.384 ± 0.007 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.326 ± 0.019 | 0.112 ± 0.008 | 0.033 ± 0.002 | 0.396 ± 0.034 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | 0.143 ± 0.007 | 0.077 ± 0.002 | 0.146 ± 0.013 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.122 ± 0.011 | 0.051 ± 0.003 | 0.136 ± 0.018 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | 0.188 ± 0.007 | 0.125 ± 0.005 | 0.221 ± 0.045 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.126 ± 0.008 | 0.051 ± 0.002 | 0.136 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.130 ± 0.008 | 0.061 ± 0.002 | 0.137 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | 0.129 ± 0.005 | 0.058 ± 0.003 | 0.339 ± 0.022 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.108 ± 0.005 | 0.034 ± 0.007 | 0.341 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | 0.172 ± 0.016 | 0.104 ± 0.002 | 0.361 ± 0.022 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.114 ± 0.016 | 0.036 ± 0.003 | 0.333 ± 0.014 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.111 ± 0.006 | 0.038 ± 0.002 | 0.350 ± 0.015 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | 0.136 ± 0.009 | 0.046 ± 0.003 | 0.064 ± 0.003 | 0.102 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | 0.281 ± 0.083 | 0.051 ± 0.006 | 0.007 ± 0.001 | 0.044 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | 0.137 ± 0.014 | 0.044 ± 0.005 | 0.221 ± 0.001 | 0.268 ± 0.013 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | 0.173 ± 0.017 | 0.045 ± 0.002 | 0.010 ± 0.000 | 0.036 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | 0.149 ± 0.015 | 0.046 ± 0.006 | 0.020 ± 0.001 | 0.062 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `16x16` | 0.140 ± 0.016 | 0.036 ± 0.006 | 0.049 ± 0.003 | 0.052 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | 0.221 ± 0.031 | 0.036 ± 0.002 | 0.022 ± 0.008 | 0.029 ± 0.011 | - | - |
| small | `qr` | f64 | 4 | `32x32` | 0.133 ± 0.005 | 0.044 ± 0.003 | 0.136 ± 0.010 | 0.161 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `4x4` | 0.137 ± 0.010 | 0.034 ± 0.005 | 0.022 ± 0.001 | 0.026 ± 0.006 | - | - |
| small | `qr` | f64 | 4 | `8x8` | 0.129 ± 0.010 | 0.034 ± 0.004 | 0.031 ± 0.002 | 0.029 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | 0.144 ± 0.004 | 0.083 ± 0.002 | 0.050 ± 0.001 | 0.075 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | 0.141 ± 0.014 | 0.081 ± 0.005 | 0.055 ± 0.001 | 0.081 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.183 ± 0.045 | 0.087 ± 0.008 | 0.014 ± 0.001 | 0.037 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.191 ± 0.025 | 0.082 ± 0.007 | 0.014 ± 0.000 | 0.041 ± 0.010 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | 0.149 ± 0.013 | 0.087 ± 0.010 | 0.135 ± 0.002 | 0.205 ± 0.075 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | 0.147 ± 0.015 | 0.081 ± 0.009 | 0.149 ± 0.002 | 0.192 ± 0.045 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.146 ± 0.006 | 0.085 ± 0.009 | 0.015 ± 0.001 | 0.039 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.150 ± 0.013 | 0.081 ± 0.007 | 0.017 ± 0.001 | 0.043 ± 0.008 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.142 ± 0.015 | 0.080 ± 0.003 | 0.021 ± 0.000 | 0.054 ± 0.013 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.144 ± 0.010 | 0.081 ± 0.008 | 0.023 ± 0.001 | 0.047 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `16x16` | 0.157 ± 0.015 | 0.056 ± 0.006 | 0.056 ± 0.001 | 0.069 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `2x2` | 0.224 ± 0.027 | 0.042 ± 0.004 | 0.008 ± 0.001 | 0.040 ± 0.016 | - | - |
| small | `svd` | f64 | 4 | `32x32` | 0.199 ± 0.015 | 0.103 ± 0.011 | 0.180 ± 0.003 | 0.213 ± 0.005 | - | - |
| small | `svd` | f64 | 4 | `4x4` | 0.160 ± 0.021 | 0.041 ± 0.005 | 0.011 ± 0.000 | 0.027 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `8x8` | 0.139 ± 0.013 | 0.042 ± 0.003 | 0.020 ± 0.001 | 0.035 ± 0.003 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 86.5x faster than the slowest successful cell (`pytorch-cpu`, 53.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 165.6x faster than the slowest successful cell (`pytorch-cpu`, 53.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 1.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 53.1x faster than the slowest successful cell (`pytorch-cpu`, 14.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 118.7x faster than the slowest successful cell (`pytorch-cpu`, 14.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 22.2x faster than the slowest successful cell (`jax-cpu`, 3.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 57.4x faster than the slowest successful cell (`jax-cpu`, 3.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 16.9x faster than the slowest successful cell (`jax-cpu`, 3.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 39.9x faster than the slowest successful cell (`jax-cpu`, 3.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 44.8x faster than the slowest successful cell (`pytorch-cpu`, 13.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 89.7x faster than the slowest successful cell (`pytorch-cpu`, 13.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 3.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 48.7x faster than the slowest successful cell (`jax-cpu`, 3.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`jax-cpu`, 1.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 33.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 33.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 9.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 2.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 41.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 41.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 10.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 10.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 38.0x faster than the slowest successful cell (`pytorch-cpu`, 57.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 81.9x faster than the slowest successful cell (`pytorch-cpu`, 57.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 1.353 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 14.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 51.2x faster than the slowest successful cell (`pytorch-cpu`, 14.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 26.5x faster than the slowest successful cell (`jax-cpu`, 3.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`jax-cpu`, 4.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 1.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 16.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 16.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 4.438 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 1.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`jax-cpu`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 43.6x faster than the slowest successful cell (`jax-cpu`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 26.4x faster than the slowest successful cell (`jax-cpu`, 3.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 1.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`1024x1024`): `tenferro-eager` is 38.4x faster than the slowest successful cell (`pytorch-cpu`, 221.597 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 46.2x faster than the slowest successful cell (`pytorch-cpu`, 221.597 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 22.5x faster than the slowest successful cell (`jax-cpu`, 3.542 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 57.0x faster than the slowest successful cell (`jax-cpu`, 3.542 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 48.6x faster than the slowest successful cell (`jax-cpu`, 13.802 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 94.9x faster than the slowest successful cell (`jax-cpu`, 13.802 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`512x512`): `tenferro-eager` is 54.6x faster than the slowest successful cell (`jax-cpu`, 53.556 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 80.6x faster than the slowest successful cell (`jax-cpu`, 53.556 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`1024x256 * 256x1024`): `tenferro-eager` is 31.9x faster than the slowest successful cell (`pytorch-cpu`, 54.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`1024x256 * 256x1024`): `tenferro-trace` is 41.3x faster than the slowest successful cell (`pytorch-cpu`, 54.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-eager` is 88.4x faster than the slowest successful cell (`jax-cpu`, 57.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-trace` is 153.0x faster than the slowest successful cell (`jax-cpu`, 57.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 1.888 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 2.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 2.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`jax-cpu`, 7.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 7.739 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 9.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/eigh` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`tenferro-eager`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 23.8x faster than the slowest successful cell (`tenferro-eager`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.380 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 0.475 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 20.4x faster than the slowest successful cell (`tenferro-eager`, 0.433 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.384 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.341 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 37.9x faster than the slowest successful cell (`tenferro-eager`, 0.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/solve` (f64, threads=4, shape=`2x2,rhs=4`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`tenferro-eager`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/svd` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

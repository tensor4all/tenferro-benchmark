# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260913_183437`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260913_183437`.

- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_183437/cpu_ops_t4_20260913_183437.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_183437/cpu_ops_t4_20260913_183437.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 11.071 ± 0.061 | 11.099 ± 0.052 | 10.078 ± 0.102 | 5.359 ± 0.079 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.186 ± 0.007 | 0.211 ± 0.009 | 0.163 ± 0.006 | 0.272 ± 0.212 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.783 ± 0.042 | 2.857 ± 0.056 | 2.554 ± 0.055 | 1.075 ± 0.033 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.701 ± 0.008 | 0.721 ± 0.016 | 0.633 ± 0.013 | 0.489 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.978 ± 0.012 | 0.985 ± 0.028 | 0.141 ± 0.010 | 0.297 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.030 ± 0.002 | 0.052 ± 0.004 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.259 ± 0.007 | 0.276 ± 0.005 | 0.035 ± 0.000 | 0.091 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.075 ± 0.002 | 0.093 ± 0.001 | 0.011 ± 0.000 | 0.038 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.558 ± 0.008 | 1.564 ± 0.028 | 0.696 ± 0.012 | 0.867 ± 0.088 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.001 | 0.060 ± 0.003 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.404 ± 0.005 | 0.418 ± 0.012 | 0.174 ± 0.004 | 0.235 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.109 ± 0.003 | 0.133 ± 0.003 | 0.046 ± 0.000 | 0.074 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.510 ± 0.027 | 3.558 ± 0.030 | 2.613 ± 0.008 | 1.133 ± 0.402 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.068 ± 0.004 | 0.089 ± 0.002 | 0.043 ± 0.000 | 0.064 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.901 ± 0.032 | 0.912 ± 0.014 | 0.656 ± 0.022 | 0.540 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.232 ± 0.004 | 0.254 ± 0.006 | 0.167 ± 0.003 | 0.194 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.312 ± 0.024 | 0.320 ± 0.025 | 1.052 ± 0.061 | 0.373 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.018 ± 0.002 | 0.043 ± 0.004 | 0.020 ± 0.002 | 0.084 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.089 ± 0.011 | 0.124 ± 0.010 | 0.267 ± 0.018 | 0.290 ± 0.050 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.031 ± 0.003 | 0.061 ± 0.003 | 0.068 ± 0.012 | 0.114 ± 0.044 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.045 ± 0.004 | 0.071 ± 0.004 | 0.009 ± 0.001 | 0.078 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.002 | 0.040 ± 0.003 | 0.005 ± 0.000 | 0.052 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.019 ± 0.002 | 0.049 ± 0.003 | 0.006 ± 0.000 | 0.069 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.041 ± 0.006 | 0.005 ± 0.000 | 0.066 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.059 ± 0.003 | 0.088 ± 0.004 | 0.033 ± 0.004 | 0.189 ± 0.030 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.002 | 0.040 ± 0.004 | 0.005 ± 0.000 | 0.069 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.023 ± 0.003 | 0.047 ± 0.003 | 0.009 ± 0.001 | 0.104 ± 0.023 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.002 | 0.041 ± 0.007 | 0.006 ± 0.000 | 0.076 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.124 ± 0.007 | 0.150 ± 0.015 | 0.498 ± 0.028 | 0.272 ± 0.038 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.002 | 0.039 ± 0.002 | 0.013 ± 0.000 | 0.072 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.040 ± 0.003 | 0.069 ± 0.007 | 0.130 ± 0.009 | 0.108 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.017 ± 0.003 | 0.045 ± 0.002 | 0.035 ± 0.003 | 0.086 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.938 ± 0.014 | 2.978 ± 0.025 | 1.789 ± 0.045 | 1.094 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.061 ± 0.003 | 0.081 ± 0.004 | 0.046 ± 0.003 | 0.054 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.748 ± 0.004 | 0.770 ± 0.013 | 0.480 ± 0.018 | 0.534 ± 0.033 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.201 ± 0.005 | 0.220 ± 0.007 | 0.134 ± 0.006 | 0.358 ± 0.037 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.024 ± 0.018 | 1.054 ± 0.015 | 0.129 ± 0.008 | 0.182 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.031 ± 0.003 | 0.051 ± 0.002 | 0.015 ± 0.003 | 0.008 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.266 ± 0.008 | 0.294 ± 0.007 | 0.046 ± 0.004 | 0.061 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.076 ± 0.001 | 0.099 ± 0.004 | 0.024 ± 0.002 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.209 ± 0.017 | 1.236 ± 0.026 | 0.281 ± 0.006 | 0.362 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.033 ± 0.004 | 0.053 ± 0.005 | 0.020 ± 0.004 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.314 ± 0.013 | 0.333 ± 0.009 | 0.085 ± 0.003 | 0.107 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.095 ± 0.005 | 0.116 ± 0.007 | 0.034 ± 0.001 | 0.041 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 1.672 ± 0.018 | 1.737 ± 0.021 | 0.696 ± 0.018 | 0.559 ± 0.085 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.040 ± 0.004 | 0.062 ± 0.004 | 0.028 ± 0.003 | 0.033 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.435 ± 0.008 | 0.459 ± 0.006 | 0.187 ± 0.005 | 0.218 ± 0.014 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.122 ± 0.006 | 0.141 ± 0.004 | 0.059 ± 0.002 | 0.069 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.074 ± 0.026 | 2.719 ± 0.026 | 0.319 ± 0.012 | 0.561 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.047 ± 0.002 | 0.092 ± 0.001 | 0.030 ± 0.005 | 0.038 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.532 ± 0.009 | 0.746 ± 0.016 | 0.115 ± 0.007 | 0.296 ± 0.022 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.147 ± 0.005 | 0.224 ± 0.005 | 0.046 ± 0.005 | 0.165 ± 0.024 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.992 ± 0.012 | 2.001 ± 0.027 | 0.067 ± 0.004 | 0.110 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.031 ± 0.003 | 0.084 ± 0.008 | 0.023 ± 0.003 | 0.007 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.263 ± 0.005 | 0.545 ± 0.013 | 0.033 ± 0.005 | 0.045 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.076 ± 0.002 | 0.178 ± 0.007 | 0.026 ± 0.004 | 0.029 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.055 ± 0.020 | 2.074 ± 0.023 | 0.082 ± 0.006 | 0.158 ± 0.017 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.032 ± 0.003 | 0.082 ± 0.002 | 0.023 ± 0.001 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.278 ± 0.003 | 0.557 ± 0.009 | 0.039 ± 0.003 | 0.056 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.083 ± 0.007 | 0.178 ± 0.002 | 0.026 ± 0.003 | 0.035 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.281 ± 0.013 | 2.218 ± 0.022 | 0.143 ± 0.005 | 0.323 ± 0.045 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.035 ± 0.002 | 0.085 ± 0.002 | 0.024 ± 0.001 | 0.024 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.341 ± 0.014 | 0.594 ± 0.015 | 0.054 ± 0.005 | 0.101 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.097 ± 0.006 | 0.187 ± 0.003 | 0.032 ± 0.003 | 0.040 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 21.900 ± 0.074 | 22.019 ± 0.059 | 20.425 ± 0.064 | 10.758 ± 0.087 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.359 ± 0.010 | 0.391 ± 0.007 | 0.321 ± 0.008 | 0.350 ± 0.149 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.499 ± 0.025 | 5.538 ± 0.023 | 5.112 ± 0.029 | 2.795 ± 0.042 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.389 ± 0.027 | 1.417 ± 0.024 | 1.285 ± 0.021 | 0.674 ± 0.090 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.596 ± 0.029 | 1.649 ± 0.037 | 0.324 ± 0.009 | 1.060 ± 0.075 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.040 ± 0.004 | 0.070 ± 0.008 | 0.008 ± 0.000 | 0.024 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.414 ± 0.009 | 0.444 ± 0.014 | 0.083 ± 0.003 | 0.284 ± 0.032 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.115 ± 0.005 | 0.145 ± 0.010 | 0.023 ± 0.001 | 0.074 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.710 ± 0.016 | 2.752 ± 0.031 | 1.409 ± 0.027 | 1.038 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.059 ± 0.004 | 0.088 ± 0.004 | 0.025 ± 0.000 | 0.040 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.692 ± 0.017 | 0.716 ± 0.009 | 0.353 ± 0.013 | 0.533 ± 0.022 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.185 ± 0.003 | 0.214 ± 0.007 | 0.090 ± 0.002 | 0.150 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.579 ± 0.025 | 6.634 ± 0.031 | 5.170 ± 0.021 | 2.104 ± 0.873 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.123 ± 0.006 | 0.149 ± 0.009 | 0.084 ± 0.001 | 0.114 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.666 ± 0.029 | 1.683 ± 0.014 | 1.295 ± 0.031 | 0.730 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.426 ± 0.013 | 0.460 ± 0.012 | 0.327 ± 0.010 | 0.469 ± 0.139 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.235 ± 0.045 | 1.066 ± 0.014 | 3.902 ± 0.118 | 0.617 ± 0.022 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.236 ± 0.014 | 0.157 ± 0.014 | 0.095 ± 0.008 | 0.042 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.496 ± 0.022 | 0.402 ± 0.015 | 1.045 ± 0.025 | 0.454 ± 0.028 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.300 ± 0.026 | 0.222 ± 0.011 | 0.288 ± 0.025 | 0.086 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.332 ± 0.012 | 0.252 ± 0.013 | 0.042 ± 0.004 | 0.031 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.219 ± 0.010 | 0.140 ± 0.010 | 0.029 ± 0.002 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.245 ± 0.008 | 0.158 ± 0.013 | 0.032 ± 0.001 | 0.022 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.236 ± 0.015 | 0.149 ± 0.006 | 0.030 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.389 ± 0.014 | 0.310 ± 0.016 | 0.114 ± 0.026 | 0.286 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.220 ± 0.014 | 0.143 ± 0.016 | 0.031 ± 0.002 | 0.023 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.254 ± 0.011 | 0.177 ± 0.016 | 0.044 ± 0.007 | 0.089 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.229 ± 0.010 | 0.153 ± 0.024 | 0.033 ± 0.003 | 0.038 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.782 ± 0.015 | 0.676 ± 0.023 | 3.350 ± 0.161 | 0.356 ± 0.077 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.231 ± 0.013 | 0.157 ± 0.013 | 0.081 ± 0.002 | 0.028 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.378 ± 0.015 | 0.309 ± 0.016 | 0.837 ± 0.032 | 0.088 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.255 ± 0.012 | 0.186 ± 0.017 | 0.236 ± 0.008 | 0.042 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 9.030 ± 0.072 | 5.150 ± 0.058 | 0.615 ± 0.031 | 0.713 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.437 ± 0.020 | 0.246 ± 0.006 | 0.053 ± 0.010 | 0.044 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.499 ± 0.052 | 1.442 ± 0.041 | 0.257 ± 0.032 | 0.343 ± 0.036 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.856 ± 0.011 | 0.491 ± 0.015 | 0.088 ± 0.010 | 0.188 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.323 ± 0.038 | 3.781 ± 0.036 | 0.124 ± 0.013 | 0.143 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.363 ± 0.012 | 0.229 ± 0.028 | 0.045 ± 0.008 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.306 ± 0.030 | 1.073 ± 0.027 | 0.065 ± 0.008 | 0.051 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.548 ± 0.020 | 0.401 ± 0.014 | 0.045 ± 0.005 | 0.031 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.613 ± 0.073 | 3.896 ± 0.034 | 0.166 ± 0.028 | 0.192 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.362 ± 0.005 | 0.225 ± 0.005 | 0.045 ± 0.005 | 0.023 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.407 ± 0.125 | 1.099 ± 0.023 | 0.078 ± 0.014 | 0.069 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.569 ± 0.033 | 0.408 ± 0.009 | 0.049 ± 0.007 | 0.035 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.615 ± 0.101 | 4.184 ± 0.049 | 0.288 ± 0.020 | 0.363 ± 0.047 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.381 ± 0.008 | 0.228 ± 0.010 | 0.045 ± 0.007 | 0.027 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.650 ± 0.037 | 1.177 ± 0.039 | 0.105 ± 0.009 | 0.112 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.632 ± 0.020 | 0.426 ± 0.010 | 0.061 ± 0.011 | 0.049 ± 0.011 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.646 ± 0.033 | 0.655 ± 0.026 | 0.621 ± 0.034 | 0.700 ± 0.037 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.756 ± 0.046 | 2.745 ± 0.039 | 2.644 ± 0.060 | 2.833 ± 0.060 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.227 ± 0.011 | 0.235 ± 0.016 | 0.202 ± 0.018 | 0.238 ± 0.017 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 74.957 ± 0.446 | 81.886 ± 0.679 | 85.585 ± 0.200 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 3.144 ± 0.052 | 3.096 ± 0.068 | 3.077 ± 0.063 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 13.854 ± 0.099 | 14.372 ± 0.086 | 15.320 ± 0.180 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 75.149 ± 0.209 | 70.606 ± 0.322 | 86.138 ± 1.170 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 3.098 ± 0.048 | 2.861 ± 0.052 | 2.898 ± 0.065 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 13.805 ± 0.093 | 13.013 ± 0.215 | 15.267 ± 0.132 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 29.543 ± 0.493 | 58.529 ± 1.211 | 35.740 ± 0.482 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.915 ± 0.094 | 1.379 ± 0.076 | 0.773 ± 0.036 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.515 ± 0.198 | 7.828 ± 0.108 | 5.385 ± 0.163 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 30.085 ± 0.380 | 30.610 ± 0.109 | 36.344 ± 0.254 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.872 ± 0.038 | 0.851 ± 0.025 | 0.832 ± 0.050 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.378 ± 0.087 | 4.420 ± 0.200 | 5.498 ± 0.192 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.052 ± 0.003 | 0.076 ± 0.005 | 0.020 ± 0.003 | 0.108 ± 0.014 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.039 ± 0.010 | 0.056 ± 0.002 | 0.005 ± 0.000 | 0.035 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.293 ± 0.017 | 0.211 ± 0.011 | 0.068 ± 0.016 | 0.215 ± 0.015 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.245 ± 0.012 | 0.155 ± 0.006 | 0.027 ± 0.003 | 0.060 ± 0.006 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 47.387 ± 1.201 | 52.356 ± 0.498 | 60.559 ± 1.943 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 2.081 ± 0.103 | 1.885 ± 0.028 | 1.805 ± 0.097 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 8.689 ± 0.060 | 8.710 ± 0.210 | 9.986 ± 0.512 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 47.865 ± 0.476 | 50.297 ± 0.729 | 64.171 ± 0.456 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 2.049 ± 0.075 | 1.829 ± 0.072 | 1.910 ± 0.033 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 8.440 ± 0.044 | 8.307 ± 0.106 | 10.466 ± 0.136 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.596 ± 0.028 | 0.242 ± 0.018 | 0.085 ± 0.016 | 0.092 ± 0.029 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.384 ± 0.017 | 0.187 ± 0.024 | 0.040 ± 0.005 | 0.046 ± 0.013 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.787 ± 0.144 | 5.568 ± 0.285 | 5.253 ± 0.155 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.385 ± 0.034 | 0.306 ± 0.022 | 0.207 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.267 ± 0.036 | 1.060 ± 0.095 | 1.163 ± 0.037 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.609 ± 0.181 | 9.452 ± 0.164 | 5.022 ± 0.330 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.382 ± 0.019 | 0.414 ± 0.014 | 0.201 ± 0.018 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.242 ± 0.026 | 1.739 ± 0.075 | 1.075 ± 0.040 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 2.650 ± 0.038 | 1.339 ± 0.022 | 1.262 ± 0.054 | 1.459 ± 0.048 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.746 ± 0.019 | 0.379 ± 0.012 | 0.304 ± 0.025 | 0.335 ± 0.029 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 119.470 ± 0.595 | 134.011 ± 0.980 | 129.422 ± 1.931 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.206 ± 0.039 | 5.620 ± 0.113 | 5.369 ± 0.073 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 23.483 ± 0.100 | 25.434 ± 0.064 | 24.566 ± 0.061 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 118.798 ± 0.781 | 114.242 ± 1.366 | 127.825 ± 0.553 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 5.168 ± 0.038 | 5.074 ± 0.097 | 5.328 ± 0.058 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 23.397 ± 0.078 | 22.842 ± 0.152 | 24.869 ± 0.247 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 5.460 ± 0.326 | 5.204 ± 0.498 | 5.099 ± 0.624 | 8.679 ± 0.067 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.063 ± 0.016 | 0.061 ± 0.005 | 0.017 ± 0.002 | 0.107 ± 0.013 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.216 ± 0.011 | 0.145 ± 0.008 | 0.090 ± 0.003 | 0.198 ± 0.012 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.947 ± 0.077 | 0.671 ± 0.033 | 0.659 ± 0.009 | 1.168 ± 0.031 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.296 ± 0.032 | 1.339 ± 0.075 | 1.299 ± 0.033 | 2.165 ± 0.058 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.358 ± 0.026 | 0.375 ± 0.023 | 0.371 ± 0.012 | 0.628 ± 0.015 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.386 ± 0.007 | 0.407 ± 0.004 | 0.396 ± 0.004 | 0.420 ± 0.020 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.179 ± 0.021 | 1.202 ± 0.018 | 1.230 ± 0.028 | 1.276 ± 0.046 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.085 ± 0.005 | 0.105 ± 0.012 | 0.090 ± 0.008 | 0.091 ± 0.010 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.085 ± 0.005 | 0.099 ± 0.006 | 0.052 ± 0.010 | 0.083 ± 0.017 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.088 ± 0.004 | 0.104 ± 0.009 | 0.051 ± 0.006 | 0.089 ± 0.014 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.111 ± 0.005 | 0.119 ± 0.013 | 0.062 ± 0.004 | 0.102 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.304 ± 0.016 | 0.249 ± 0.014 | 0.190 ± 0.010 | 0.239 ± 0.020 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.317 ± 0.017 | 0.251 ± 0.008 | 0.191 ± 0.019 | 0.241 ± 0.014 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.353 ± 0.015 | 0.302 ± 0.013 | 0.225 ± 0.016 | 0.287 ± 0.024 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.039 ± 0.008 | 0.059 ± 0.010 | 0.019 ± 0.002 | 0.041 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.038 ± 0.004 | 0.066 ± 0.005 | 0.020 ± 0.001 | 0.042 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.046 ± 0.002 | 0.075 ± 0.006 | 0.024 ± 0.002 | 0.050 ± 0.010 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.248 ± 0.017 | 1.273 ± 0.020 | 1.234 ± 0.033 | 1.270 ± 0.035 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.890 ± 0.024 | 4.992 ± 0.059 | 4.883 ± 0.033 | 4.954 ± 0.036 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.305 ± 0.014 | 0.332 ± 0.018 | 0.283 ± 0.016 | 0.364 ± 0.040 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_eigh` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 2.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 30.0x faster than the slowest successful cell (`tenferro-trace`, 2.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 0.545 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`tenferro-trace`, 0.545 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 2.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 2.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 0.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`tenferro-trace`, 2.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 0.594 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 38.1x faster than the slowest successful cell (`tenferro-eager`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 9.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 9.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 30.2x faster than the slowest successful cell (`tenferro-eager`, 4.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 34.8x faster than the slowest successful cell (`tenferro-eager`, 4.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 39.2x faster than the slowest successful cell (`tenferro-eager`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 1.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 1.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-eager`, 4.613 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 27.8x faster than the slowest successful cell (`tenferro-eager`, 4.613 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.362 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 1.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 1.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 5.615 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 5.615 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 1.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 1.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

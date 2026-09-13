# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260913_153332`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260913_153332`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_153332/cpu_ops_t4_20260913_153332.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_153332/cpu_ops_t4_20260913_153332.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 11.040 ± 0.547 | 11.010 ± 0.050 | 10.222 ± 0.089 | 5.381 ± 0.052 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.196 ± 0.004 | 0.196 ± 0.007 | 0.162 ± 0.007 | 0.280 ± 0.146 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.780 ± 0.031 | 2.780 ± 0.026 | 2.575 ± 0.031 | 1.238 ± 0.317 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.713 ± 0.006 | 0.705 ± 0.007 | 0.647 ± 0.015 | 0.478 ± 0.058 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.984 ± 0.019 | 0.979 ± 0.021 | 0.148 ± 0.014 | 0.286 ± 0.022 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.036 ± 0.002 | 0.040 ± 0.002 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.264 ± 0.006 | 0.266 ± 0.007 | 0.037 ± 0.002 | 0.090 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.083 ± 0.004 | 0.084 ± 0.002 | 0.011 ± 0.000 | 0.042 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.563 ± 0.015 | 1.550 ± 0.015 | 0.713 ± 0.018 | 0.651 ± 0.184 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.044 ± 0.001 | 0.049 ± 0.003 | 0.014 ± 0.000 | 0.020 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.407 ± 0.004 | 0.406 ± 0.008 | 0.177 ± 0.007 | 0.245 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.118 ± 0.003 | 0.120 ± 0.002 | 0.046 ± 0.002 | 0.077 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.528 ± 0.055 | 3.500 ± 0.030 | 2.639 ± 0.051 | 1.522 ± 0.408 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.079 ± 0.004 | 0.079 ± 0.002 | 0.044 ± 0.002 | 0.065 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.896 ± 0.008 | 0.880 ± 0.014 | 0.665 ± 0.014 | 0.571 ± 0.034 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.241 ± 0.011 | 0.240 ± 0.003 | 0.167 ± 0.009 | 0.201 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.293 ± 0.013 | 0.336 ± 0.035 | 1.064 ± 0.049 | 0.343 ± 0.027 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.035 ± 0.006 | 0.035 ± 0.002 | 0.022 ± 0.003 | 0.130 ± 0.052 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.108 ± 0.006 | 0.108 ± 0.007 | 0.281 ± 0.019 | 0.306 ± 0.029 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.050 ± 0.002 | 0.050 ± 0.006 | 0.077 ± 0.016 | 0.114 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.058 ± 0.002 | 0.063 ± 0.003 | 0.009 ± 0.001 | 0.082 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.004 | 0.031 ± 0.003 | 0.005 ± 0.000 | 0.052 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.037 ± 0.003 | 0.039 ± 0.003 | 0.006 ± 0.000 | 0.068 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.003 | 0.033 ± 0.002 | 0.005 ± 0.000 | 0.063 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.074 ± 0.003 | 0.075 ± 0.002 | 0.038 ± 0.006 | 0.189 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.030 ± 0.003 | 0.032 ± 0.003 | 0.005 ± 0.000 | 0.074 ± 0.035 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.036 ± 0.002 | 0.039 ± 0.002 | 0.009 ± 0.002 | 0.112 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.004 | 0.035 ± 0.002 | 0.006 ± 0.001 | 0.082 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.138 ± 0.002 | 0.144 ± 0.008 | 0.533 ± 0.014 | 0.247 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.033 ± 0.007 | 0.034 ± 0.003 | 0.013 ± 0.001 | 0.091 ± 0.082 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.060 ± 0.007 | 0.061 ± 0.003 | 0.130 ± 0.015 | 0.117 ± 0.038 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.034 ± 0.012 | 0.037 ± 0.003 | 0.037 ± 0.003 | 0.086 ± 0.029 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.965 ± 0.059 | 2.891 ± 0.026 | 1.833 ± 0.030 | 1.132 ± 0.036 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.070 ± 0.007 | 0.069 ± 0.003 | 0.049 ± 0.006 | 0.058 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.750 ± 0.016 | 0.741 ± 0.023 | 0.482 ± 0.009 | 0.461 ± 0.068 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.208 ± 0.007 | 0.204 ± 0.006 | 0.135 ± 0.006 | 0.321 ± 0.033 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.033 ± 0.016 | 1.028 ± 0.007 | 0.136 ± 0.009 | 0.182 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.037 ± 0.004 | 0.040 ± 0.001 | 0.016 ± 0.006 | 0.009 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.275 ± 0.006 | 0.282 ± 0.005 | 0.047 ± 0.003 | 0.061 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.005 | 0.088 ± 0.003 | 0.023 ± 0.002 | 0.017 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.213 ± 0.011 | 1.229 ± 0.019 | 0.285 ± 0.011 | 0.367 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.039 ± 0.002 | 0.043 ± 0.002 | 0.020 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.322 ± 0.007 | 0.322 ± 0.007 | 0.087 ± 0.009 | 0.109 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.098 ± 0.003 | 0.099 ± 0.003 | 0.035 ± 0.001 | 0.043 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 1.678 ± 0.018 | 1.676 ± 0.020 | 0.705 ± 0.023 | 0.494 ± 0.066 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.047 ± 0.004 | 0.050 ± 0.002 | 0.027 ± 0.004 | 0.035 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.441 ± 0.004 | 0.432 ± 0.008 | 0.187 ± 0.013 | 0.223 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.124 ± 0.003 | 0.128 ± 0.002 | 0.061 ± 0.002 | 0.071 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.100 ± 0.041 | 2.711 ± 0.037 | 0.336 ± 0.014 | 0.574 ± 0.032 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.062 ± 0.003 | 0.105 ± 0.006 | 0.032 ± 0.009 | 0.040 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.557 ± 0.039 | 0.730 ± 0.018 | 0.125 ± 0.013 | 0.271 ± 0.033 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.169 ± 0.004 | 0.237 ± 0.010 | 0.048 ± 0.005 | 0.159 ± 0.034 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 1.025 ± 0.012 | 1.993 ± 0.031 | 0.076 ± 0.006 | 0.107 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.045 ± 0.004 | 0.100 ± 0.011 | 0.024 ± 0.005 | 0.008 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.280 ± 0.006 | 0.555 ± 0.006 | 0.034 ± 0.002 | 0.047 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.096 ± 0.009 | 0.194 ± 0.004 | 0.025 ± 0.002 | 0.030 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.092 ± 0.012 | 2.046 ± 0.027 | 0.098 ± 0.014 | 0.153 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.045 ± 0.005 | 0.100 ± 0.008 | 0.023 ± 0.003 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.298 ± 0.012 | 0.559 ± 0.011 | 0.039 ± 0.005 | 0.057 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.097 ± 0.007 | 0.196 ± 0.003 | 0.025 ± 0.003 | 0.030 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.325 ± 0.009 | 2.210 ± 0.036 | 0.156 ± 0.012 | 0.231 ± 0.044 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.054 ± 0.005 | 0.102 ± 0.006 | 0.025 ± 0.002 | 0.029 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.363 ± 0.009 | 0.606 ± 0.009 | 0.060 ± 0.008 | 0.112 ± 0.021 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.112 ± 0.006 | 0.212 ± 0.018 | 0.033 ± 0.004 | 0.042 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 22.001 ± 0.581 | 21.783 ± 0.121 | 20.624 ± 0.159 | 10.906 ± 0.495 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.364 ± 0.007 | 0.364 ± 0.007 | 0.328 ± 0.014 | 0.249 ± 0.148 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.470 ± 0.067 | 5.514 ± 0.054 | 5.188 ± 0.025 | 2.836 ± 0.047 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.390 ± 0.024 | 1.384 ± 0.014 | 1.305 ± 0.045 | 0.678 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.582 ± 0.021 | 1.580 ± 0.021 | 0.331 ± 0.017 | 1.072 ± 0.067 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.047 ± 0.001 | 0.051 ± 0.003 | 0.009 ± 0.000 | 0.024 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.415 ± 0.009 | 0.416 ± 0.007 | 0.084 ± 0.004 | 0.298 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.123 ± 0.005 | 0.123 ± 0.003 | 0.024 ± 0.001 | 0.077 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.676 ± 0.023 | 2.693 ± 0.042 | 1.418 ± 0.035 | 0.883 ± 0.174 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.064 ± 0.005 | 0.067 ± 0.003 | 0.026 ± 0.001 | 0.040 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.693 ± 0.010 | 0.687 ± 0.017 | 0.357 ± 0.016 | 0.566 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.191 ± 0.007 | 0.192 ± 0.003 | 0.092 ± 0.003 | 0.156 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.573 ± 0.030 | 6.518 ± 0.063 | 5.248 ± 0.048 | 3.066 ± 1.112 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.134 ± 0.008 | 0.130 ± 0.007 | 0.085 ± 0.002 | 0.115 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.659 ± 0.015 | 1.658 ± 0.018 | 1.313 ± 0.044 | 0.719 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.435 ± 0.006 | 0.438 ± 0.006 | 0.333 ± 0.016 | 0.432 ± 0.088 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.153 ± 0.028 | 1.021 ± 0.035 | 3.849 ± 0.117 | 0.627 ± 0.067 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.219 ± 0.009 | 0.137 ± 0.008 | 0.104 ± 0.007 | 0.047 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.492 ± 0.053 | 0.393 ± 0.014 | 1.058 ± 0.035 | 0.435 ± 0.033 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.264 ± 0.016 | 0.192 ± 0.006 | 0.290 ± 0.038 | 0.086 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.299 ± 0.014 | 0.229 ± 0.005 | 0.045 ± 0.009 | 0.031 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.190 ± 0.013 | 0.126 ± 0.011 | 0.029 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.223 ± 0.012 | 0.148 ± 0.014 | 0.034 ± 0.002 | 0.021 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.207 ± 0.008 | 0.126 ± 0.008 | 0.031 ± 0.003 | 0.021 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.347 ± 0.006 | 0.279 ± 0.007 | 0.118 ± 0.028 | 0.288 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.220 ± 0.068 | 0.119 ± 0.002 | 0.031 ± 0.001 | 0.025 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.222 ± 0.008 | 0.155 ± 0.005 | 0.048 ± 0.014 | 0.089 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.203 ± 0.008 | 0.128 ± 0.006 | 0.033 ± 0.001 | 0.038 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.751 ± 0.027 | 0.641 ± 0.013 | 3.312 ± 0.273 | 0.261 ± 0.030 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.213 ± 0.020 | 0.130 ± 0.007 | 0.082 ± 0.001 | 0.024 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.341 ± 0.009 | 0.271 ± 0.004 | 0.886 ± 0.036 | 0.113 ± 0.021 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.228 ± 0.012 | 0.166 ± 0.017 | 0.249 ± 0.020 | 0.042 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 8.831 ± 0.059 | 5.072 ± 0.033 | 0.626 ± 0.024 | 0.731 ± 0.079 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.425 ± 0.022 | 0.287 ± 0.009 | 0.062 ± 0.017 | 0.045 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.480 ± 0.029 | 1.484 ± 0.022 | 0.249 ± 0.044 | 0.312 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.847 ± 0.022 | 0.538 ± 0.022 | 0.099 ± 0.021 | 0.173 ± 0.025 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.306 ± 0.059 | 3.744 ± 0.044 | 0.160 ± 0.067 | 0.145 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.356 ± 0.016 | 0.273 ± 0.014 | 0.047 ± 0.011 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.296 ± 0.011 | 1.104 ± 0.012 | 0.065 ± 0.010 | 0.057 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.536 ± 0.009 | 0.448 ± 0.015 | 0.048 ± 0.006 | 0.032 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.608 ± 0.021 | 3.914 ± 0.075 | 0.183 ± 0.033 | 0.198 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.354 ± 0.041 | 0.275 ± 0.019 | 0.044 ± 0.004 | 0.030 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.367 ± 0.011 | 1.138 ± 0.030 | 0.087 ± 0.022 | 0.071 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.555 ± 0.015 | 0.448 ± 0.011 | 0.050 ± 0.011 | 0.037 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.600 ± 0.023 | 4.152 ± 0.041 | 0.348 ± 0.038 | 0.291 ± 0.028 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.366 ± 0.012 | 0.278 ± 0.010 | 0.047 ± 0.005 | 0.038 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.620 ± 0.023 | 1.201 ± 0.014 | 0.117 ± 0.030 | 0.134 ± 0.024 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.623 ± 0.027 | 0.474 ± 0.016 | 0.064 ± 0.007 | 0.047 ± 0.008 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.632 ± 0.007 | 0.646 ± 0.023 | 0.618 ± 0.019 | 0.702 ± 0.036 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.701 ± 0.029 | 2.692 ± 0.016 | 2.618 ± 0.042 | 2.885 ± 0.108 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.227 ± 0.009 | 0.225 ± 0.009 | 0.198 ± 0.017 | 0.244 ± 0.026 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 73.590 ± 0.219 | 83.368 ± 1.314 | 86.838 ± 0.538 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 3.057 ± 0.022 | 3.188 ± 0.101 | 3.212 ± 0.082 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 13.592 ± 0.085 | 14.652 ± 0.189 | 15.515 ± 0.174 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 74.085 ± 0.178 | 72.067 ± 1.547 | 89.216 ± 9.486 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 3.035 ± 0.049 | 2.942 ± 0.118 | 3.153 ± 0.121 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 13.568 ± 0.065 | 13.965 ± 1.299 | 15.534 ± 0.140 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 27.850 ± 1.022 | 60.333 ± 4.310 | 36.831 ± 0.624 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.902 ± 0.043 | 1.434 ± 0.067 | 0.826 ± 0.045 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.367 ± 0.177 | 9.384 ± 0.794 | 5.570 ± 0.147 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 28.442 ± 0.881 | 31.705 ± 2.578 | 37.488 ± 0.535 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.850 ± 0.023 | 0.869 ± 0.026 | 0.851 ± 0.082 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.287 ± 0.061 | 5.301 ± 0.140 | 5.593 ± 0.263 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.050 ± 0.001 | 0.067 ± 0.003 | 0.018 ± 0.001 | 0.120 ± 0.009 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.038 ± 0.002 | 0.047 ± 0.003 | 0.006 ± 0.001 | 0.039 ± 0.008 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.251 ± 0.009 | 0.184 ± 0.003 | 0.058 ± 0.006 | 0.246 ± 0.027 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.217 ± 0.015 | 0.135 ± 0.012 | 0.025 ± 0.003 | 0.058 ± 0.008 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 47.447 ± 6.380 | 57.834 ± 1.792 | 59.783 ± 1.612 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 2.026 ± 0.056 | 1.930 ± 0.125 | 1.920 ± 0.133 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 8.460 ± 0.058 | 8.973 ± 0.112 | 10.367 ± 0.190 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 46.627 ± 0.379 | 53.242 ± 7.321 | 65.626 ± 0.405 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 2.003 ± 0.013 | 1.902 ± 0.119 | 1.964 ± 0.064 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 8.314 ± 0.058 | 8.711 ± 0.158 | 10.722 ± 0.207 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.579 ± 0.010 | 0.288 ± 0.015 | 0.113 ± 0.024 | 0.102 ± 0.019 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.366 ± 0.021 | 0.233 ± 0.010 | 0.042 ± 0.005 | 0.046 ± 0.015 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.794 ± 0.305 | 5.838 ± 0.360 | 5.801 ± 0.316 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.415 ± 0.015 | 0.323 ± 0.084 | 0.245 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.286 ± 0.028 | 1.329 ± 0.315 | 1.216 ± 0.061 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.460 ± 0.165 | 9.748 ± 0.295 | 5.515 ± 0.206 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.404 ± 0.007 | 0.423 ± 0.039 | 0.230 ± 0.012 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.234 ± 0.015 | 2.020 ± 0.342 | 1.099 ± 0.047 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 2.607 ± 0.022 | 1.318 ± 0.016 | 1.283 ± 0.057 | 1.496 ± 0.059 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.736 ± 0.013 | 0.366 ± 0.011 | 0.294 ± 0.004 | 0.341 ± 0.031 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 117.026 ± 0.273 | 145.150 ± 5.372 | 133.247 ± 0.965 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.101 ± 0.030 | 5.795 ± 0.104 | 5.532 ± 0.086 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 23.141 ± 0.101 | 26.187 ± 0.178 | 25.173 ± 0.193 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 116.038 ± 0.745 | 117.445 ± 1.779 | 132.491 ± 0.209 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 5.053 ± 0.014 | 5.296 ± 0.078 | 5.455 ± 0.100 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 22.955 ± 0.032 | 23.626 ± 0.097 | 25.059 ± 0.104 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.909 ± 0.508 | 4.952 ± 0.525 | 4.877 ± 0.648 | 8.866 ± 0.092 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.096 ± 0.042 | 0.047 ± 0.003 | 0.015 ± 0.001 | 0.129 ± 0.035 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.238 ± 0.008 | 0.134 ± 0.006 | 0.078 ± 0.001 | 0.222 ± 0.029 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.903 ± 0.122 | 0.667 ± 0.007 | 0.633 ± 0.019 | 1.212 ± 0.105 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.302 ± 0.020 | 1.308 ± 0.008 | 1.263 ± 0.021 | 2.300 ± 0.173 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.368 ± 0.014 | 0.374 ± 0.008 | 0.350 ± 0.010 | 0.648 ± 0.065 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.386 ± 0.005 | 0.391 ± 0.008 | 0.391 ± 0.009 | 0.433 ± 0.025 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.180 ± 0.011 | 1.180 ± 0.015 | 1.214 ± 0.017 | 1.306 ± 0.048 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.090 ± 0.006 | 0.089 ± 0.004 | 0.086 ± 0.006 | 0.094 ± 0.012 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.099 ± 0.005 | 0.121 ± 0.012 | 0.048 ± 0.002 | 0.094 ± 0.023 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.103 ± 0.007 | 0.123 ± 0.006 | 0.049 ± 0.002 | 0.095 ± 0.021 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.126 ± 0.005 | 0.143 ± 0.008 | 0.060 ± 0.002 | 0.114 ± 0.011 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.319 ± 0.011 | 0.266 ± 0.012 | 0.176 ± 0.002 | 0.274 ± 0.014 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.329 ± 0.005 | 0.268 ± 0.011 | 0.180 ± 0.004 | 0.276 ± 0.020 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.379 ± 0.005 | 0.321 ± 0.014 | 0.213 ± 0.008 | 0.316 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.052 ± 0.008 | 0.086 ± 0.007 | 0.018 ± 0.001 | 0.043 ± 0.010 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.051 ± 0.002 | 0.085 ± 0.006 | 0.019 ± 0.001 | 0.045 ± 0.009 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.061 ± 0.004 | 0.092 ± 0.006 | 0.025 ± 0.004 | 0.051 ± 0.012 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.230 ± 0.007 | 1.242 ± 0.015 | 1.223 ± 0.023 | 1.348 ± 0.084 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.834 ± 0.022 | 4.841 ± 0.013 | 4.809 ± 0.021 | 5.078 ± 0.104 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.310 ± 0.017 | 0.306 ± 0.018 | 0.282 ± 0.007 | 0.371 ± 0.031 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-trace`, 1.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 1.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 0.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 0.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 2.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 20.9x faster than the slowest successful cell (`tenferro-trace`, 2.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 0.559 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 2.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 0.606 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 34.3x faster than the slowest successful cell (`tenferro-eager`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 3.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 8.831 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 8.831 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 29.7x faster than the slowest successful cell (`tenferro-eager`, 4.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 4.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 38.1x faster than the slowest successful cell (`tenferro-eager`, 0.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 1.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 1.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 4.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 25.2x faster than the slowest successful cell (`tenferro-eager`, 4.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`tenferro-eager`, 1.367 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 1.367 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`tenferro-eager`, 5.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 5.600 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 1.620 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 1.620 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

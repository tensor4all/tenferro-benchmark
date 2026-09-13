# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260913_152441`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260913_152441`.

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

## Threads: 1

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_152441/cpu_ops_t1_20260913_152441.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_152441/cpu_ops_t1_20260913_152441.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 11.241 ± 0.062 | 11.258 ± 0.097 | 10.261 ± 0.079 | 10.399 ± 0.054 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.177 ± 0.004 | 0.179 ± 0.002 | 0.164 ± 0.008 | 0.183 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.810 ± 0.078 | 2.827 ± 0.064 | 2.596 ± 0.048 | 2.595 ± 0.038 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.705 ± 0.018 | 0.719 ± 0.021 | 0.654 ± 0.019 | 0.667 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.986 ± 0.033 | 0.984 ± 0.020 | 0.147 ± 0.016 | 0.277 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.022 ± 0.001 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.247 ± 0.007 | 0.247 ± 0.004 | 0.036 ± 0.001 | 0.087 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.067 ± 0.002 | 0.066 ± 0.000 | 0.012 ± 0.001 | 0.037 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.577 ± 0.046 | 1.589 ± 0.032 | 0.711 ± 0.029 | 0.832 ± 0.021 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.031 ± 0.001 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.397 ± 0.012 | 0.398 ± 0.012 | 0.176 ± 0.005 | 0.230 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.104 ± 0.003 | 0.104 ± 0.005 | 0.046 ± 0.002 | 0.072 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 3.541 ± 0.061 | 3.586 ± 0.061 | 2.670 ± 0.035 | 2.721 ± 0.058 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.059 ± 0.003 | 0.060 ± 0.003 | 0.044 ± 0.002 | 0.064 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.898 ± 0.024 | 0.896 ± 0.024 | 0.682 ± 0.041 | 0.700 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.224 ± 0.006 | 0.222 ± 0.002 | 0.169 ± 0.011 | 0.195 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.348 ± 0.008 | 0.292 ± 0.014 | 1.117 ± 0.050 | 0.318 ± 0.036 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.011 ± 0.001 | 0.012 ± 0.000 | 0.021 ± 0.002 | 0.081 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.083 ± 0.005 | 0.081 ± 0.004 | 0.296 ± 0.025 | 0.126 ± 0.024 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.025 ± 0.001 | 0.026 ± 0.000 | 0.080 ± 0.010 | 0.092 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.038 ± 0.001 | 0.039 ± 0.001 | 0.009 ± 0.002 | 0.075 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.048 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.001 | 0.062 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.060 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.053 ± 0.001 | 0.052 ± 0.001 | 0.028 ± 0.008 | 0.088 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.060 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | 0.077 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.063 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.115 ± 0.007 | 0.117 ± 0.005 | 0.532 ± 0.027 | 0.130 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.066 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.036 ± 0.007 | 0.135 ± 0.010 | 0.094 ± 0.014 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.001 | 0.038 ± 0.002 | 0.081 ± 0.034 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 2.998 ± 0.046 | 2.933 ± 0.027 | 1.944 ± 0.043 | 2.023 ± 0.036 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.052 ± 0.001 | 0.052 ± 0.004 | 0.031 ± 0.001 | 0.054 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.753 ± 0.026 | 0.749 ± 0.021 | 0.490 ± 0.020 | 0.524 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.192 ± 0.004 | 0.192 ± 0.006 | 0.121 ± 0.006 | 0.150 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.029 ± 0.020 | 1.029 ± 0.023 | 0.127 ± 0.013 | 0.175 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.023 ± 0.001 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.263 ± 0.008 | 0.263 ± 0.018 | 0.032 ± 0.001 | 0.062 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.069 ± 0.001 | 0.069 ± 0.003 | 0.010 ± 0.000 | 0.017 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.202 ± 0.039 | 1.218 ± 0.053 | 0.284 ± 0.012 | 0.337 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.001 | 0.025 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.304 ± 0.006 | 0.310 ± 0.004 | 0.073 ± 0.004 | 0.105 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.080 ± 0.003 | 0.081 ± 0.001 | 0.020 ± 0.001 | 0.039 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 1.704 ± 0.038 | 1.681 ± 0.072 | 0.717 ± 0.023 | 0.769 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.031 ± 0.000 | 0.033 ± 0.002 | 0.013 ± 0.000 | 0.031 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.435 ± 0.023 | 0.432 ± 0.017 | 0.175 ± 0.005 | 0.218 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.111 ± 0.002 | 0.112 ± 0.007 | 0.045 ± 0.001 | 0.072 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.121 ± 0.045 | 2.778 ± 0.040 | 0.646 ± 0.013 | 0.760 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.040 ± 0.001 | 0.062 ± 0.003 | 0.016 ± 0.002 | 0.037 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.536 ± 0.015 | 0.712 ± 0.013 | 0.191 ± 0.018 | 0.201 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.139 ± 0.007 | 0.191 ± 0.004 | 0.049 ± 0.011 | 0.074 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 1.008 ± 0.015 | 2.052 ± 0.024 | 0.079 ± 0.016 | 0.098 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.023 ± 0.001 | 0.050 ± 0.002 | 0.007 ± 0.000 | 0.007 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.251 ± 0.003 | 0.521 ± 0.006 | 0.021 ± 0.002 | 0.044 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.069 ± 0.003 | 0.145 ± 0.003 | 0.010 ± 0.000 | 0.024 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.069 ± 0.027 | 2.114 ± 0.044 | 0.119 ± 0.018 | 0.136 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.024 ± 0.000 | 0.051 ± 0.004 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.275 ± 0.004 | 0.536 ± 0.011 | 0.030 ± 0.002 | 0.053 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.074 ± 0.003 | 0.152 ± 0.007 | 0.012 ± 0.000 | 0.034 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.302 ± 0.018 | 2.263 ± 0.062 | 0.260 ± 0.012 | 0.265 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.054 ± 0.005 | 0.010 ± 0.001 | 0.026 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.339 ± 0.002 | 0.581 ± 0.027 | 0.066 ± 0.008 | 0.087 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.088 ± 0.001 | 0.154 ± 0.006 | 0.020 ± 0.001 | 0.041 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 22.309 ± 0.303 | 22.239 ± 0.194 | 20.719 ± 0.181 | 20.977 ± 0.235 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.352 ± 0.008 | 0.354 ± 0.021 | 0.321 ± 0.016 | 0.341 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.600 ± 0.083 | 5.570 ± 0.048 | 5.211 ± 0.074 | 5.251 ± 0.036 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.417 ± 0.037 | 1.403 ± 0.048 | 1.298 ± 0.052 | 1.322 ± 0.030 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.622 ± 0.051 | 1.613 ± 0.043 | 0.333 ± 0.017 | 1.034 ± 0.040 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.032 ± 0.001 | 0.033 ± 0.001 | 0.009 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.399 ± 0.008 | 0.406 ± 0.004 | 0.082 ± 0.006 | 0.284 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.107 ± 0.009 | 0.107 ± 0.002 | 0.024 ± 0.000 | 0.074 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.743 ± 0.058 | 2.757 ± 0.051 | 1.444 ± 0.060 | 2.034 ± 0.043 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.049 ± 0.001 | 0.049 ± 0.004 | 0.025 ± 0.001 | 0.040 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.698 ± 0.024 | 0.699 ± 0.035 | 0.357 ± 0.021 | 0.525 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.175 ± 0.006 | 0.178 ± 0.007 | 0.092 ± 0.005 | 0.148 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 6.717 ± 0.070 | 6.718 ± 0.077 | 5.270 ± 0.075 | 5.687 ± 0.077 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.111 ± 0.001 | 0.112 ± 0.003 | 0.084 ± 0.006 | 0.111 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.698 ± 0.024 | 1.683 ± 0.044 | 1.323 ± 0.044 | 1.481 ± 0.047 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.424 ± 0.007 | 0.427 ± 0.012 | 0.330 ± 0.013 | 0.373 ± 0.018 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.229 ± 0.080 | 1.045 ± 0.052 | 4.229 ± 0.181 | 0.908 ± 0.027 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.102 ± 0.003 | 0.042 ± 0.001 | 0.097 ± 0.007 | 0.039 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.356 ± 0.028 | 0.259 ± 0.009 | 1.149 ± 0.059 | 0.238 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.150 ± 0.007 | 0.087 ± 0.018 | 0.323 ± 0.066 | 0.082 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.208 ± 0.009 | 0.126 ± 0.002 | 0.045 ± 0.025 | 0.030 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.071 ± 0.001 | 0.024 ± 0.001 | 0.030 ± 0.002 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.104 ± 0.003 | 0.048 ± 0.001 | 0.033 ± 0.004 | 0.020 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.078 ± 0.003 | 0.027 ± 0.002 | 0.031 ± 0.004 | 0.019 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.260 ± 0.007 | 0.170 ± 0.005 | 0.114 ± 0.033 | 0.196 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.072 ± 0.003 | 0.024 ± 0.001 | 0.031 ± 0.004 | 0.023 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.118 ± 0.002 | 0.060 ± 0.000 | 0.047 ± 0.019 | 0.066 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.082 ± 0.001 | 0.032 ± 0.000 | 0.034 ± 0.002 | 0.031 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.659 ± 0.037 | 0.572 ± 0.038 | 3.576 ± 0.077 | 0.243 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.081 ± 0.002 | 0.034 ± 0.006 | 0.083 ± 0.008 | 0.026 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.225 ± 0.011 | 0.169 ± 0.004 | 0.906 ± 0.035 | 0.083 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.117 ± 0.010 | 0.063 ± 0.003 | 0.252 ± 0.029 | 0.041 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 9.306 ± 0.141 | 5.250 ± 0.112 | 1.020 ± 0.029 | 0.912 ± 0.042 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.244 ± 0.008 | 0.133 ± 0.003 | 0.039 ± 0.006 | 0.043 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.396 ± 0.049 | 1.363 ± 0.039 | 0.349 ± 0.037 | 0.237 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.675 ± 0.027 | 0.380 ± 0.007 | 0.096 ± 0.021 | 0.084 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.278 ± 0.073 | 3.793 ± 0.046 | 0.147 ± 0.022 | 0.125 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.167 ± 0.005 | 0.111 ± 0.000 | 0.025 ± 0.001 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.146 ± 0.036 | 0.999 ± 0.042 | 0.050 ± 0.006 | 0.052 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.366 ± 0.014 | 0.285 ± 0.004 | 0.029 ± 0.001 | 0.030 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.590 ± 0.063 | 3.946 ± 0.076 | 0.229 ± 0.045 | 0.179 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.175 ± 0.013 | 0.112 ± 0.004 | 0.026 ± 0.002 | 0.023 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.225 ± 0.029 | 1.022 ± 0.033 | 0.063 ± 0.014 | 0.066 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.387 ± 0.012 | 0.300 ± 0.015 | 0.033 ± 0.004 | 0.036 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.677 ± 0.095 | 4.191 ± 0.048 | 0.428 ± 0.034 | 0.318 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.188 ± 0.004 | 0.116 ± 0.004 | 0.028 ± 0.001 | 0.027 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.499 ± 0.038 | 1.094 ± 0.037 | 0.120 ± 0.026 | 0.106 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.449 ± 0.019 | 0.312 ± 0.008 | 0.045 ± 0.003 | 0.047 ± 0.007 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.630 ± 0.029 | 0.623 ± 0.030 | 0.625 ± 0.034 | 0.694 ± 0.022 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.773 ± 0.035 | 2.748 ± 0.048 | 2.743 ± 0.056 | 2.844 ± 0.043 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.216 ± 0.022 | 0.206 ± 0.008 | 0.202 ± 0.012 | 0.232 ± 0.016 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 77.401 ± 0.538 | 85.895 ± 1.288 | 134.321 ± 0.431 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 3.066 ± 0.085 | 3.127 ± 0.048 | 3.940 ± 0.036 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 14.166 ± 0.165 | 14.912 ± 0.118 | 21.484 ± 0.074 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 78.018 ± 1.330 | 73.734 ± 1.017 | 132.915 ± 0.248 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 3.023 ± 0.049 | 2.867 ± 0.064 | 3.919 ± 0.038 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 14.087 ± 0.106 | 13.476 ± 0.187 | 21.250 ± 0.093 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 31.655 ± 0.977 | 70.085 ± 3.526 | 86.173 ± 0.636 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.723 ± 0.052 | 1.341 ± 0.028 | 1.735 ± 0.010 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 4.376 ± 0.245 | 9.112 ± 0.252 | 11.947 ± 0.054 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 31.369 ± 0.523 | 35.086 ± 1.339 | 86.883 ± 0.263 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.719 ± 0.033 | 0.779 ± 0.040 | 1.744 ± 0.021 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 4.483 ± 0.172 | 5.051 ± 0.096 | 11.793 ± 0.096 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.030 ± 0.003 | 0.028 ± 0.005 | 0.024 ± 0.004 | 0.100 ± 0.006 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.006 ± 0.001 | 0.033 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.144 ± 0.005 | 0.089 ± 0.007 | 0.086 ± 0.036 | 0.248 ± 0.008 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.093 ± 0.024 | 0.037 ± 0.001 | 0.027 ± 0.006 | 0.053 ± 0.004 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 53.897 ± 0.697 | 60.776 ± 1.281 | 134.588 ± 0.353 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.922 ± 0.105 | 1.945 ± 0.049 | 3.154 ± 0.013 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 9.347 ± 0.080 | 10.397 ± 0.186 | 18.824 ± 0.180 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 54.259 ± 1.244 | 61.842 ± 0.756 | 136.882 ± 0.356 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.912 ± 0.094 | 1.891 ± 0.087 | 3.130 ± 0.009 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 9.261 ± 0.044 | 9.913 ± 0.167 | 19.482 ± 0.069 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.425 ± 0.019 | 0.133 ± 0.011 | 0.090 ± 0.012 | 0.090 ± 0.013 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.189 ± 0.010 | 0.074 ± 0.001 | 0.038 ± 0.003 | 0.044 ± 0.008 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 6.209 ± 0.081 | 6.399 ± 0.191 | 5.185 ± 0.069 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.293 ± 0.007 | 0.320 ± 0.068 | 0.244 ± 0.016 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.250 ± 0.026 | 1.284 ± 0.108 | 1.010 ± 0.012 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 6.131 ± 0.194 | 10.562 ± 0.157 | 4.728 ± 0.043 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.295 ± 0.013 | 0.383 ± 0.005 | 0.236 ± 0.007 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.240 ± 0.034 | 1.967 ± 0.065 | 0.986 ± 0.016 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 2.618 ± 0.053 | 1.280 ± 0.043 | 1.318 ± 0.098 | 1.415 ± 0.032 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.635 ± 0.040 | 0.304 ± 0.008 | 0.297 ± 0.008 | 0.338 ± 0.021 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 121.893 ± 0.605 | 142.589 ± 1.694 | 177.237 ± 1.042 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.201 ± 0.097 | 5.738 ± 0.138 | 6.079 ± 0.042 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 23.931 ± 0.094 | 26.508 ± 0.086 | 30.890 ± 0.059 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 120.411 ± 0.766 | 117.949 ± 1.181 | 176.587 ± 0.472 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 5.179 ± 0.099 | 5.173 ± 0.078 | 6.022 ± 0.021 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 23.777 ± 0.073 | 23.156 ± 0.105 | 31.077 ± 0.589 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 5.892 ± 0.855 | 5.890 ± 0.334 | 5.871 ± 0.829 | 33.043 ± 0.219 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.040 ± 0.002 | 0.022 ± 0.000 | 0.018 ± 0.005 | 0.128 ± 0.025 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.193 ± 0.073 | 0.107 ± 0.004 | 0.101 ± 0.004 | 0.577 ± 0.026 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.987 ± 0.097 | 0.719 ± 0.053 | 0.656 ± 0.035 | 4.245 ± 0.086 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.338 ± 0.036 | 1.341 ± 0.045 | 1.343 ± 0.076 | 8.129 ± 0.059 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.410 ± 0.030 | 0.393 ± 0.012 | 0.379 ± 0.016 | 2.163 ± 0.024 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.386 ± 0.024 | 0.387 ± 0.009 | 0.415 ± 0.019 | 0.409 ± 0.011 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.169 ± 0.033 | 1.188 ± 0.033 | 1.406 ± 0.018 | 1.252 ± 0.021 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.072 ± 0.001 | 0.074 ± 0.001 | 0.069 ± 0.002 | 0.095 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.079 ± 0.009 | 0.070 ± 0.001 | 0.054 ± 0.006 | 0.078 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.081 ± 0.002 | 0.077 ± 0.002 | 0.055 ± 0.007 | 0.083 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.105 ± 0.003 | 0.098 ± 0.005 | 0.067 ± 0.005 | 0.100 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.321 ± 0.022 | 0.212 ± 0.005 | 0.216 ± 0.031 | 0.222 ± 0.014 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.327 ± 0.011 | 0.215 ± 0.003 | 0.230 ± 0.031 | 0.222 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.379 ± 0.012 | 0.268 ± 0.013 | 0.261 ± 0.017 | 0.264 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.026 ± 0.002 | 0.034 ± 0.002 | 0.018 ± 0.002 | 0.040 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.030 ± 0.002 | 0.036 ± 0.000 | 0.020 ± 0.006 | 0.043 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.039 ± 0.003 | 0.044 ± 0.007 | 0.025 ± 0.004 | 0.049 ± 0.007 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.253 ± 0.045 | 1.239 ± 0.055 | 1.249 ± 0.037 | 1.272 ± 0.018 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.961 ± 0.080 | 4.965 ± 0.062 | 5.014 ± 0.075 | 4.901 ± 0.039 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.287 ± 0.018 | 0.280 ± 0.003 | 0.285 ± 0.028 | 0.356 ± 0.025 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-trace`, 2.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 26.1x faster than the slowest successful cell (`tenferro-trace`, 2.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 0.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 25.1x faster than the slowest successful cell (`tenferro-trace`, 0.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-trace`, 2.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 2.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 0.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 0.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 3.576 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 9.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 2.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 34.4x faster than the slowest successful cell (`tenferro-eager`, 4.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 29.1x faster than the slowest successful cell (`tenferro-eager`, 4.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 1.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 23.1x faster than the slowest successful cell (`tenferro-eager`, 1.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 25.7x faster than the slowest successful cell (`tenferro-eager`, 4.590 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 4.590 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 1.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 1.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.387 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 5.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 5.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 1.499 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 1.499 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.449 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

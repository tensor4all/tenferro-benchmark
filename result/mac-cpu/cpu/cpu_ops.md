# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260812_171837`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260812_171837`.

- tenferro-rs commit: `3ce8fbed00c2aae40ba55ca38f04e32527c2116e`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260812_171837/cpu_ops_t4_20260812_171837.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260812_171837/cpu_ops_t4_20260812_171837.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.028 ± 0.000 | 0.022 ± 0.001 | 0.011 ± 0.000 | 0.035 ± 0.009 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.018 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.064 ± 0.001 | 0.055 ± 0.003 | 0.046 ± 0.000 | 0.075 ± 0.011 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.002 ± 0.001 | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.051 ± 0.005 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.080 ± 0.032 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.065 ± 0.002 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.071 ± 0.008 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.004 | 0.008 ± 0.001 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.027 ± 0.000 | 0.023 ± 0.002 | 0.023 ± 0.003 | 0.017 ± 0.001 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.002 | 0.011 ± 0.001 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.038 ± 0.001 | 0.035 ± 0.001 | 0.035 ± 0.003 | 0.046 ± 0.014 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.012 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.001 | 0.007 ± 0.001 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.032 ± 0.003 | 0.028 ± 0.002 | 0.025 ± 0.001 | 0.026 ± 0.006 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.001 | 0.014 ± 0.013 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.038 ± 0.000 | 0.030 ± 0.002 | 0.025 ± 0.002 | 0.036 ± 0.008 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.001 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.046 ± 0.001 | 0.041 ± 0.001 | 0.023 ± 0.001 | 0.078 ± 0.003 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.032 ± 0.000 | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.040 ± 0.002 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.110 ± 0.000 | 0.109 ± 0.000 | 0.090 ± 0.000 | 0.200 ± 0.066 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.074 ± 0.011 | 0.036 ± 0.007 | 0.027 ± 0.001 | 0.382 ± 0.029 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.070 ± 0.003 | 0.036 ± 0.004 | 0.029 ± 0.001 | 0.396 ± 0.050 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.074 ± 0.004 | 0.044 ± 0.004 | 0.029 ± 0.001 | 1.009 ± 0.494 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.071 ± 0.003 | 0.043 ± 0.003 | 0.032 ± 0.001 | 0.429 ± 0.036 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.133 ± 0.006 | 0.091 ± 0.005 | 0.040 ± 0.008 | 0.324 ± 0.008 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.146 ± 0.012 | 0.131 ± 0.007 | 0.044 ± 0.002 | 0.372 ± 0.091 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.128 ± 0.007 | 0.093 ± 0.002 | 0.043 ± 0.006 | 0.494 ± 0.174 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.154 ± 0.012 | 0.143 ± 0.011 | 0.046 ± 0.005 | 0.409 ± 0.300 |
| large | `eigh` | f64 | 4 | `64x64` | 0.306 ± 0.003 | 0.200 ± 0.003 | 0.201 ± 0.002 | 0.233 ± 0.016 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.028 ± 0.004 | 0.015 ± 0.006 | 0.006 ± 0.000 | 0.034 ± 0.004 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.103 ± 0.003 | 0.044 ± 0.001 | 0.026 ± 0.011 | 0.328 ± 0.056 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.183 ± 0.013 | 0.096 ± 0.005 | 0.039 ± 0.002 | 0.339 ± 0.017 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.699 ± 0.022 | 0.319 ± 0.015 | 0.302 ± 0.018 | 0.468 ± 0.046 |
| large | `matmul` | f64 | 4 | `128x128` | 0.028 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.001 | 0.094 ± 0.008 |
| large | `matmul` | f64 | 4 | `256x256` | 0.153 ± 0.000 | 0.093 ± 0.002 | 0.077 ± 0.002 | 0.182 ± 0.030 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.506 ± 0.033 | 0.456 ± 0.017 | 0.361 ± 0.007 | 0.422 ± 0.053 |
| large | `qr` | f64 | 4 | `64x64` | 0.086 ± 0.000 | 0.066 ± 0.002 | 0.089 ± 0.006 | 0.093 ± 0.013 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.083 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.003 | 0.037 ± 0.005 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.086 ± 0.001 | 0.020 ± 0.001 | 0.019 ± 0.000 | 0.045 ± 0.008 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.099 ± 0.006 | 0.030 ± 0.003 | 0.026 ± 0.005 | 0.049 ± 0.002 |
| large | `svd` | f64 | 4 | `64x64` | 0.359 ± 0.028 | 0.274 ± 0.002 | 0.282 ± 0.004 | 0.312 ± 0.039 |
| small | `eigh` | f64 | 4 | `2x2` | 0.004 ± 0.003 | 0.001 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 |
| small | `eigh` | f64 | 4 | `4x4` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 |
| small | `eigh` | f64 | 4 | `8x8` | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.001 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.005 ± 0.002 | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.048 ± 0.003 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | 0.051 ± 0.003 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | 0.060 ± 0.007 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.147 ± 0.023 | 0.036 ± 0.004 | 0.012 ± 0.002 | 0.283 ± 0.009 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.114 ± 0.006 | 0.033 ± 0.003 | 0.013 ± 0.001 | 0.295 ± 0.018 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.097 ± 0.011 | 0.036 ± 0.002 | 0.015 ± 0.001 | 0.301 ± 0.018 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.188 ± 0.013 | 0.067 ± 0.006 | 0.019 ± 0.001 | 0.322 ± 0.009 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.156 ± 0.009 | 0.068 ± 0.006 | 0.020 ± 0.000 | 0.331 ± 0.017 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.149 ± 0.011 | 0.069 ± 0.007 | 0.020 ± 0.000 | 0.327 ± 0.016 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.546 ± 0.018 | 0.034 ± 0.004 | 0.014 ± 0.001 | 0.350 ± 0.006 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.452 ± 0.007 | 0.033 ± 0.002 | 0.016 ± 0.000 | 0.359 ± 0.016 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.450 ± 0.018 | 0.039 ± 0.006 | 0.020 ± 0.001 | 0.371 ± 0.027 |
| small | `matmul` | f64 | 4 | `2x2` | 0.006 ± 0.003 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 |
| small | `matmul` | f64 | 4 | `4x4` | 0.003 ± 0.001 | 0.001 ± 0.001 | 0.001 ± 0.000 | 0.005 ± 0.000 |
| small | `matmul` | f64 | 4 | `8x8` | 0.003 ± 0.002 | 0.001 ± 0.000 | 0.001 ± 0.000 | 0.019 ± 0.005 |
| small | `qr` | f64 | 4 | `2x2` | 0.004 ± 0.002 | 0.001 ± 0.000 | 0.014 ± 0.003 | 0.006 ± 0.001 |
| small | `qr` | f64 | 4 | `4x4` | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.017 ± 0.005 | 0.006 ± 0.001 |
| small | `qr` | f64 | 4 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.018 ± 0.003 | 0.007 ± 0.001 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.008 ± 0.002 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.001 |
| small | `svd` | f64 | 4 | `2x2` | 0.005 ± 0.002 | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 |
| small | `svd` | f64 | 4 | `4x4` | 0.006 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 |
| small | `svd` | f64 | 4 | `8x8` | 0.011 ± 0.001 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.001 |

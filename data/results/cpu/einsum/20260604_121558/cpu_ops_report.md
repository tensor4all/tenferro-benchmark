# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Timestamp: `20260604_121558`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/cpu/einsum/20260604_121558`.

- tenferro-rs commit: `ba6523f81e3fb35b00ba3b3f96b551ed95cd6d1b`

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
- Python platform: `macOS-26.5-arm64-arm-64bit`

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

## Python Backend BLAS Providers

- PyTorch: provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: provider `internal_lapack`, version `0.10.1`, backend `cpu`
  - linked BLAS/LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 1

- CSV: `data/results/cpu/einsum/20260604_121558/cpu_ops_t1_20260604_121558.csv`
- Source table: `data/results/cpu/einsum/20260604_121558/cpu_ops_t1_20260604_121558.md`

## PR884 CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | Torch C++ (ms) | PyTorch Python (ms) | JAX Python (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.011 ± 0.000 | 0.009 ± 0.000 | - | 0.111 ± 0.002 | 0.144 ± 0.012 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.036 ± 0.000 | 0.030 ± 0.000 | - | 0.393 ± 0.002 | 0.488 ± 0.119 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.025 ± 0.000 | 0.022 ± 0.000 | - | 0.158 ± 0.004 | 0.199 ± 0.005 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.094 ± 0.000 | 0.085 ± 0.000 | - | 0.608 ± 0.009 | 0.705 ± 0.039 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.037 ± 0.001 | 0.162 ± 0.010 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.005 ± 0.000 | 0.003 ± 0.000 | - | 0.095 ± 0.001 | 0.252 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.003 ± 0.000 | 0.002 ± 0.000 | - | 0.096 ± 0.005 | 0.237 ± 0.017 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.006 ± 0.000 | 0.004 ± 0.000 | - | 0.346 ± 0.006 | 0.484 ± 0.013 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.013 ± 0.000 | 0.011 ± 0.000 | - | 0.074 ± 0.001 | 0.106 ± 0.009 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.043 ± 0.000 | 0.040 ± 0.000 | - | 0.263 ± 0.004 | 0.309 ± 0.011 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.017 ± 0.000 | 0.015 ± 0.000 | - | 0.113 ± 0.001 | 0.148 ± 0.012 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.061 ± 0.000 | 0.056 ± 0.000 | - | 0.425 ± 0.006 | 0.512 ± 0.037 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.009 ± 0.000 | 0.006 ± 0.000 | - | 0.122 ± 0.002 | 0.178 ± 0.011 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.028 ± 0.000 | 0.022 ± 0.000 | - | 0.425 ± 0.010 | 0.519 ± 0.014 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.011 ± 0.000 | 0.007 ± 0.000 | - | 0.162 ± 0.002 | 0.222 ± 0.003 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.036 ± 0.000 | 0.026 ± 0.000 | - | 0.606 ± 0.008 | 0.705 ± 0.019 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.017 ± 0.000 | 0.016 ± 0.000 | - | 0.082 ± 0.003 | 0.129 ± 0.011 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.060 ± 0.000 | 0.058 ± 0.000 | - | 0.287 ± 0.001 | 0.402 ± 0.012 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.043 ± 0.000 | 0.041 ± 0.000 | - | 0.142 ± 0.004 | 0.200 ± 0.015 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.161 ± 0.001 | 0.163 ± 0.006 | - | 0.536 ± 0.010 | 0.665 ± 0.036 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.037 ± 0.001 | 0.008 ± 0.000 | - | 0.075 ± 0.004 | 0.709 ± 0.064 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.043 ± 0.001 | 0.015 ± 0.000 | - | 0.134 ± 0.005 | 0.774 ± 0.068 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.039 ± 0.000 | 0.009 ± 0.000 | - | 0.134 ± 0.005 | 0.781 ± 0.064 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.046 ± 0.000 | 0.020 ± 0.001 | - | 0.372 ± 0.007 | 1.055 ± 0.089 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.053 ± 0.001 | 0.041 ± 0.000 | - | 0.155 ± 0.004 | 0.712 ± 0.034 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.090 ± 0.000 | 0.127 ± 0.001 | - | 0.457 ± 0.002 | 1.042 ± 0.046 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.055 ± 0.001 | 0.046 ± 0.000 | - | 0.194 ± 0.003 | 0.780 ± 0.050 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.103 ± 0.000 | 0.147 ± 0.001 | - | 0.630 ± 0.007 | 1.225 ± 0.092 |
| large | `eigh` | f64 | 1 | `64x64` | 0.470 ± 0.004 | 0.396 ± 0.002 | - | 0.971 ± 0.091 | 1.179 ± 0.157 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.016 ± 0.000 | 0.008 ± 0.001 | - | 1.269 ± 0.021 | 1.437 ± 0.096 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.054 ± 0.000 | 0.025 ± 0.000 | - | 1.320 ± 0.018 | 2.027 ± 0.205 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.144 ± 0.000 | 0.100 ± 0.003 | - | 0.742 ± 0.029 | 1.329 ± 0.153 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.827 ± 0.003 | 0.795 ± 0.020 | - | 1.107 ± 0.054 | 1.440 ± 0.235 |
| large | `matmul` | f64 | 1 | `128x128` | 0.034 ± 0.000 | 0.023 ± 0.000 | - | 4.930 ± 0.078 | 5.346 ± 0.100 |
| large | `matmul` | f64 | 1 | `256x256` | 0.193 ± 0.002 | 0.163 ± 0.011 | - | 19.612 ± 0.187 | 20.883 ± 0.427 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.656 ± 0.002 | 0.670 ± 0.013 | - | 79.528 ± 1.829 | 84.105 ± 2.175 |
| large | `qr` | f64 | 1 | `64x64` | 0.105 ± 0.002 | 0.098 ± 0.002 | - | 0.751 ± 0.036 | 0.843 ± 0.021 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.090 ± 0.000 | 0.017 ± 0.000 | - | 0.693 ± 0.011 | 0.835 ± 0.074 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.090 ± 0.000 | 0.020 ± 0.002 | - | 0.845 ± 0.004 | 1.005 ± 0.050 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.097 ± 0.008 | 0.028 ± 0.002 | - | 1.325 ± 0.027 | 1.478 ± 0.032 |
| large | `svd` | f64 | 1 | `64x64` | 0.775 ± 0.002 | 0.770 ± 0.003 | - | 1.077 ± 0.028 | 1.169 ± 0.010 |
| small | `eigh` | f64 | 1 | `2x2` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.006 | 0.040 ± 0.005 |
| small | `eigh` | f64 | 1 | `4x4` | 0.005 ± 0.000 | 0.002 ± 0.000 | - | 0.016 ± 0.000 | 0.048 ± 0.010 |
| small | `eigh` | f64 | 1 | `8x8` | 0.007 ± 0.000 | 0.004 ± 0.000 | - | 0.026 ± 0.002 | 0.060 ± 0.009 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.037 ± 0.005 | 0.002 ± 0.000 | - | 0.018 ± 0.001 | 0.134 ± 0.008 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.029 ± 0.003 | 0.002 ± 0.000 | - | 0.022 ± 0.001 | 0.146 ± 0.010 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.024 ± 0.001 | 0.002 ± 0.000 | - | 0.036 ± 0.003 | 0.164 ± 0.011 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.034 ± 0.003 | 0.005 ± 0.000 | - | 0.032 ± 0.005 | 0.542 ± 0.026 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.047 ± 0.001 | 0.005 ± 0.000 | - | 0.034 ± 0.001 | 0.531 ± 0.039 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.047 ± 0.001 | 0.007 ± 0.000 | - | 0.054 ± 0.001 | 0.552 ± 0.030 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.054 ± 0.001 | 0.012 ± 0.000 | - | 0.048 ± 0.002 | 0.591 ± 0.039 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.047 ± 0.001 | 0.013 ± 0.000 | - | 0.050 ± 0.001 | 0.580 ± 0.032 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.047 ± 0.001 | 0.014 ± 0.000 | - | 0.057 ± 0.001 | 0.600 ± 0.028 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.077 ± 0.001 | 0.011 ± 0.000 | - | 0.031 ± 0.002 | 0.572 ± 0.032 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.068 ± 0.001 | 0.012 ± 0.000 | - | 0.034 ± 0.001 | 0.585 ± 0.031 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.082 ± 0.001 | 0.023 ± 0.000 | - | 0.048 ± 0.001 | 0.610 ± 0.024 |
| small | `matmul` | f64 | 1 | `2x2` | 0.008 ± 0.006 | 0.001 ± 0.000 | - | 0.012 ± 0.002 | 0.059 ± 0.007 |
| small | `matmul` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.015 ± 0.001 | 0.062 ± 0.008 |
| small | `matmul` | f64 | 1 | `8x8` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.029 ± 0.001 | 0.092 ± 0.004 |
| small | `qr` | f64 | 1 | `2x2` | 0.002 ± 0.000 | 0.001 ± 0.000 | - | 0.011 ± 0.002 | 0.038 ± 0.006 |
| small | `qr` | f64 | 1 | `4x4` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.012 ± 0.000 | 0.043 ± 0.011 |
| small | `qr` | f64 | 1 | `8x8` | 0.004 ± 0.000 | 0.002 ± 0.000 | - | 0.020 ± 0.000 | 0.051 ± 0.007 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.023 ± 0.001 | 0.064 ± 0.005 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.003 ± 0.000 | 0.001 ± 0.000 | - | 0.023 ± 0.000 | 0.065 ± 0.010 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.025 ± 0.001 | 0.070 ± 0.008 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.026 ± 0.001 | 0.068 ± 0.006 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.033 ± 0.001 | 0.074 ± 0.009 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.004 ± 0.000 | 0.001 ± 0.000 | - | 0.036 ± 0.000 | 0.082 ± 0.005 |
| small | `svd` | f64 | 1 | `2x2` | 0.003 ± 0.002 | 0.002 ± 0.000 | - | 0.013 ± 0.004 | 0.043 ± 0.012 |
| small | `svd` | f64 | 1 | `4x4` | 0.007 ± 0.000 | 0.003 ± 0.000 | - | 0.016 ± 0.000 | 0.047 ± 0.012 |
| small | `svd` | f64 | 1 | `8x8` | 0.012 ± 0.001 | 0.008 ± 0.000 | - | 0.029 ± 0.001 | 0.061 ± 0.005 |

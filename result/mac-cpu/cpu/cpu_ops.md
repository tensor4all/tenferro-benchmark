# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261123_172252/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_172252/run.yaml`
- Timestamp: `2026-11-23T17:22:52.769897+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261123_172252`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261123_172252/cpu_ops_t1_20261123_172252.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.040 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.029 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.001 | 0.082 ± 0.001 | 0.081 ± 0.000 | 0.102 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.066 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.084 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.038 ± 0.000 | 0.037 ± 0.000 | 0.036 ± 0.001 | 0.097 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.086 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.089 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.025 ± 0.002 | 0.026 ± 0.000 | 0.036 ± 0.000 | 0.044 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.008 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.015 ± 0.003 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.037 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.034 ± 0.001 | 0.035 ± 0.000 | 0.039 ± 0.000 | 0.121 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.045 ± 0.001 | 0.068 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.132 ± 0.000 | 0.137 ± 0.001 | 0.152 ± 0.001 | 0.232 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.034 ± 0.001 | 0.008 ± 0.000 | 0.068 ± 0.002 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.038 ± 0.001 | 0.014 ± 0.000 | 0.057 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.034 ± 0.000 | 0.009 ± 0.000 | 0.061 ± 0.002 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.041 ± 0.000 | 0.017 ± 0.000 | 0.061 ± 0.003 | 0.027 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.045 ± 0.000 | 0.026 ± 0.000 | 0.055 ± 0.003 | 0.012 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.049 ± 0.001 | 0.046 ± 0.001 | 0.055 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.046 ± 0.001 | 0.031 ± 0.000 | 0.048 ± 0.001 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.055 ± 0.001 | 0.060 ± 0.001 | 0.061 ± 0.001 | 0.027 ± 0.000 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.254 ± 0.004 | 0.255 ± 0.001 | 0.281 ± 0.010 | 0.291 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 20.839 ± 0.265 | 10.627 ± 0.173 | 4.963 ± 0.075 | 6.256 ± 0.103 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 116.245 ± 2.519 | 61.065 ± 0.609 | 27.377 ± 0.100 | 37.995 ± 0.674 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 14.473 ± 0.007 | 10.445 ± 0.094 | 4.605 ± 0.132 | 6.257 ± 0.072 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 92.209 ± 2.618 | 61.028 ± 0.616 | 25.144 ± 1.105 | 37.067 ± 0.449 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 10.367 ± 0.031 | 4.717 ± 0.080 | 1.919 ± 0.181 | 2.634 ± 0.011 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 70.834 ± 0.795 | 34.628 ± 0.200 | 14.283 ± 0.175 | 17.080 ± 0.034 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 9.475 ± 0.028 | 4.721 ± 0.019 | 1.189 ± 0.082 | 2.551 ± 0.019 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 67.773 ± 0.409 | 34.176 ± 1.360 | 7.958 ± 0.222 | 16.797 ± 0.058 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.024 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | 0.027 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.083 ± 0.000 | 0.053 ± 0.000 | 0.045 ± 0.000 | 0.064 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 14.877 ± 0.039 | 7.753 ± 0.021 | 2.490 ± 0.093 | 4.365 ± 0.039 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 102.102 ± 0.833 | 50.800 ± 0.344 | 13.222 ± 0.136 | 27.898 ± 0.396 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 12.996 ± 0.215 | 7.789 ± 0.097 | 2.375 ± 0.064 | 4.287 ± 0.060 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 91.736 ± 1.007 | 50.280 ± 0.266 | 12.674 ± 0.331 | 28.415 ± 0.897 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.061 ± 0.001 | 0.078 ± 0.001 | 0.065 ± 0.004 | 0.034 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.629 ± 0.031 | 0.937 ± 0.001 | 0.425 ± 0.007 | 0.288 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 9.368 ± 0.237 | 5.860 ± 0.104 | 1.682 ± 0.048 | 1.452 ± 0.045 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.072 ± 0.010 | 0.939 ± 0.005 | 0.544 ± 0.024 | 0.281 ± 0.007 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 6.283 ± 0.094 | 5.767 ± 0.066 | 2.687 ± 0.070 | 1.426 ± 0.013 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.586 ± 0.002 | 0.796 ± 0.007 | 0.441 ± 0.050 | 0.444 ± 0.007 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 38.935 ± 0.134 | 21.032 ± 0.163 | 9.500 ± 0.072 | 10.234 ± 0.047 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 233.182 ± 9.516 | 123.334 ± 0.629 | 54.181 ± 0.778 | 62.560 ± 8.535 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 26.384 ± 0.058 | 21.138 ± 0.046 | 8.617 ± 0.077 | 10.370 ± 0.147 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 163.539 ± 10.863 | 123.453 ± 0.535 | 47.611 ± 2.319 | 60.589 ± 3.006 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.106 ± 0.002 | 0.109 ± 0.005 | 0.025 ± 0.002 | 0.118 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.837 ± 0.002 | 0.843 ± 0.009 | 0.151 ± 0.030 | 0.791 ± 0.037 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.107 ± 0.019 | 3.157 ± 0.062 | 0.603 ± 0.024 | 3.362 ± 0.079 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.086 ± 0.001 | 0.096 ± 0.001 | 0.063 ± 0.001 | 0.077 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.019 ± 0.001 | 0.021 ± 0.000 | 0.034 ± 0.003 | 0.028 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.037 ± 0.001 | 0.041 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.054 ± 0.000 | 0.043 ± 0.000 | 0.069 ± 0.036 | 0.102 ± 0.013 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.535 ± 0.005 | 0.561 ± 0.006 | 0.396 ± 0.009 | 0.424 ± 0.045 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.065 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.063 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.085 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.101 ± 0.001 | 0.016 ± 0.000 | 0.076 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.109 ± 0.003 | 0.017 ± 0.000 | 0.069 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.118 ± 0.002 | 0.026 ± 0.001 | 0.077 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.037 ± 0.000 | 0.016 ± 0.000 | 0.067 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.041 ± 0.001 | 0.017 ± 0.002 | 0.059 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.048 ± 0.001 | 0.025 ± 0.001 | 0.067 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.333 ± 0.003 | 0.018 ± 0.000 | 0.113 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.351 ± 0.002 | 0.020 ± 0.001 | 0.102 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.354 ± 0.012 | 0.021 ± 0.001 | 0.106 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.082 ± 0.001 | 0.016 ± 0.000 | 0.091 ± 0.008 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.086 ± 0.001 | 0.018 ± 0.001 | 0.081 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.088 ± 0.002 | 0.019 ± 0.001 | 0.084 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.032 ± 0.002 | 0.006 ± 0.000 | 0.024 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.031 ± 0.001 | 0.006 ± 0.000 | 0.025 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.031 ± 0.001 | 0.007 ± 0.000 | 0.029 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.444 ± 0.003 | 0.024 ± 0.000 | 0.086 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.460 ± 0.002 | 0.024 ± 0.000 | 0.078 ± 0.003 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.462 ± 0.012 | 0.030 ± 0.000 | 0.081 ± 0.004 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.075 ± 0.001 | 0.025 ± 0.000 | 0.085 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.077 ± 0.000 | 0.024 ± 0.000 | 0.076 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.081 ± 0.001 | 0.028 ± 0.001 | 0.080 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.042 ± 0.002 | 0.019 ± 0.000 | 0.039 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.042 ± 0.001 | 0.019 ± 0.000 | 0.041 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.042 ± 0.015 | 0.022 ± 0.001 | 0.039 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.076 ± 0.001 | 0.014 ± 0.000 | 0.166 ± 0.022 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.079 ± 0.002 | 0.015 ± 0.000 | 0.152 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.081 ± 0.007 | 0.016 ± 0.000 | 0.146 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.042 ± 0.001 | 0.014 ± 0.000 | 0.075 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.044 ± 0.001 | 0.015 ± 0.000 | 0.076 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.045 ± 0.000 | 0.016 ± 0.000 | 0.072 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.037 ± 0.001 | 0.011 ± 0.000 | 0.030 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.037 ± 0.000 | 0.013 ± 0.000 | 0.033 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.043 ± 0.001 | 0.023 ± 0.000 | 0.039 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.093 ± 0.003 | 0.011 ± 0.000 | 0.095 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.099 ± 0.005 | 0.014 ± 0.000 | 0.100 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.115 ± 0.001 | 0.024 ± 0.001 | 0.106 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.039 ± 0.003 | 0.011 ± 0.000 | 0.072 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.043 ± 0.001 | 0.013 ± 0.000 | 0.064 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.054 ± 0.001 | 0.024 ± 0.001 | 0.069 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 26.8x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.5x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 33.8x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 32.3x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 20.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 28.3x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 37.1x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 50.5x faster than the slowest successful cell (`tenferro-eager`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 53.4x faster than the slowest successful cell (`tenferro-eager`, 0.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 0.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 75.8x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 55.1x faster than the slowest successful cell (`tenferro-eager`, 0.444 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 0.444 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 62.7x faster than the slowest successful cell (`tenferro-eager`, 0.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 0.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 94.1x faster than the slowest successful cell (`tenferro-eager`, 0.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.2x faster than the slowest successful cell (`tenferro-eager`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261123_174752/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261123_174752/run.yaml`
- Timestamp: `2026-11-23T17:47:52.092097+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261123_174752`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261123_174752/cpu_ops_t4_20261123_174752.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.003 | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.029 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.086 ± 0.014 | 0.079 ± 0.001 | 0.082 ± 0.001 | 0.113 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.067 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.086 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.039 ± 0.001 | 0.045 ± 0.001 | 0.064 ± 0.007 | 0.181 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.086 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.081 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.034 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.049 ± 0.002 | 0.028 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.040 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.001 | 0.025 ± 0.000 | 0.064 ± 0.002 | 0.052 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.032 ± 0.003 | 0.008 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.052 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.045 ± 0.003 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.013 ± 0.000 | 0.056 ± 0.001 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.039 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.036 ± 0.001 | 0.035 ± 0.000 | 0.042 ± 0.000 | 0.121 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.040 ± 0.001 | 0.045 ± 0.002 | 0.067 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.139 ± 0.002 | 0.132 ± 0.000 | 0.163 ± 0.004 | 0.241 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.040 ± 0.005 | 0.012 ± 0.000 | 0.064 ± 0.001 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.048 ± 0.003 | 0.015 ± 0.000 | 0.062 ± 0.002 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.011 ± 0.000 | 0.059 ± 0.001 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.060 ± 0.006 | 0.018 ± 0.000 | 0.060 ± 0.001 | 0.044 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.052 ± 0.000 | 0.042 ± 0.001 | 0.078 ± 0.003 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.061 ± 0.001 | 0.053 ± 0.000 | 0.104 ± 0.002 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.054 ± 0.001 | 0.042 ± 0.002 | 0.085 ± 0.008 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.069 ± 0.014 | 0.067 ± 0.000 | 0.109 ± 0.002 | 0.033 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.258 ± 0.003 | 0.265 ± 0.002 | 0.292 ± 0.004 | 0.326 ± 0.024 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 21.653 ± 1.542 | 10.311 ± 0.075 | 5.123 ± 0.038 | 5.971 ± 0.084 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 112.687 ± 4.279 | 58.949 ± 5.635 | 28.979 ± 0.225 | 29.619 ± 0.310 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 14.817 ± 0.116 | 10.095 ± 0.046 | 4.714 ± 0.042 | 5.974 ± 0.218 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 77.532 ± 0.459 | 57.911 ± 0.066 | 26.320 ± 0.233 | 29.857 ± 0.347 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 8.050 ± 0.097 | 3.495 ± 0.011 | 2.389 ± 0.072 | 1.539 ± 0.175 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 43.276 ± 0.344 | 21.060 ± 0.564 | 14.802 ± 0.076 | 8.966 ± 0.556 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 7.331 ± 0.237 | 3.527 ± 0.019 | 1.629 ± 0.026 | 1.384 ± 0.013 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 39.934 ± 0.237 | 21.954 ± 0.947 | 8.396 ± 0.697 | 8.550 ± 0.111 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.030 ± 0.002 | 0.021 ± 0.000 | 0.010 ± 0.000 | 0.032 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.167 ± 0.004 | 0.054 ± 0.001 | 0.047 ± 0.003 | 0.069 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 12.176 ± 0.083 | 6.637 ± 0.060 | 2.733 ± 0.055 | 3.460 ± 0.106 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 65.073 ± 0.886 | 32.872 ± 0.420 | 13.020 ± 0.121 | 15.173 ± 0.328 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.396 ± 0.071 | 6.493 ± 0.052 | 2.590 ± 0.047 | 3.603 ± 0.216 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 54.482 ± 0.264 | 36.044 ± 3.591 | 12.428 ± 0.028 | 15.664 ± 0.222 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.075 ± 0.001 | 0.089 ± 0.001 | 0.070 ± 0.001 | 0.050 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.853 ± 0.086 | 0.983 ± 0.009 | 0.500 ± 0.009 | 0.303 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 7.659 ± 0.152 | 5.363 ± 0.078 | 1.713 ± 0.396 | 1.436 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.239 ± 0.111 | 0.981 ± 0.012 | 0.653 ± 0.010 | 0.294 ± 0.003 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 5.271 ± 0.140 | 5.304 ± 0.201 | 2.818 ± 0.158 | 1.320 ± 0.023 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.684 ± 0.011 | 0.841 ± 0.007 | 0.415 ± 0.005 | 0.453 ± 0.014 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 40.338 ± 0.228 | 20.526 ± 0.152 | 9.752 ± 0.076 | 9.521 ± 0.173 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 232.829 ± 0.863 | 117.547 ± 0.616 | 52.058 ± 2.035 | 49.334 ± 0.351 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 26.733 ± 0.249 | 20.444 ± 0.138 | 8.737 ± 0.136 | 10.798 ± 0.243 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 159.608 ± 5.473 | 117.652 ± 0.593 | 46.259 ± 0.901 | 52.476 ± 0.651 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.088 ± 0.002 | 0.086 ± 0.002 | 0.022 ± 0.001 | 0.102 ± 0.006 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.478 ± 0.005 | 0.461 ± 0.003 | 0.167 ± 0.005 | 0.508 ± 0.108 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.658 ± 0.013 | 1.687 ± 0.034 | 0.566 ± 0.007 | 1.378 ± 0.031 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.089 ± 0.000 | 0.090 ± 0.001 | 0.103 ± 0.004 | 0.073 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.035 ± 0.004 | 0.036 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.029 ± 0.000 | 0.028 ± 0.000 | 0.040 ± 0.003 | 0.050 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.058 ± 0.000 | 0.045 ± 0.000 | 0.046 ± 0.007 | 0.111 ± 0.006 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.544 ± 0.006 | 0.543 ± 0.005 | 0.378 ± 0.003 | 0.414 ± 0.009 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.067 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.067 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.085 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.118 ± 0.001 | 0.023 ± 0.000 | 0.064 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.115 ± 0.002 | 0.024 ± 0.000 | 0.066 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.130 ± 0.002 | 0.032 ± 0.000 | 0.090 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.052 ± 0.003 | 0.023 ± 0.000 | 0.055 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.051 ± 0.000 | 0.023 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.061 ± 0.001 | 0.030 ± 0.000 | 0.079 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.386 ± 0.011 | 0.027 ± 0.000 | 0.235 ± 0.008 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.363 ± 0.003 | 0.026 ± 0.000 | 0.284 ± 0.014 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.382 ± 0.002 | 0.028 ± 0.000 | 0.261 ± 0.010 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.114 ± 0.001 | 0.024 ± 0.000 | 0.177 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.111 ± 0.001 | 0.025 ± 0.003 | 0.212 ± 0.014 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.119 ± 0.005 | 0.026 ± 0.001 | 0.207 ± 0.013 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.040 ± 0.002 | 0.010 ± 0.000 | 0.023 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.037 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.038 ± 0.001 | 0.009 ± 0.000 | 0.029 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.495 ± 0.012 | 0.036 ± 0.000 | 0.119 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.472 ± 0.001 | 0.035 ± 0.003 | 0.120 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.505 ± 0.012 | 0.039 ± 0.000 | 0.146 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.106 ± 0.001 | 0.036 ± 0.001 | 0.120 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.102 ± 0.001 | 0.035 ± 0.000 | 0.121 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.001 | 0.039 ± 0.001 | 0.154 ± 0.012 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.052 ± 0.001 | 0.030 ± 0.004 | 0.038 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.049 ± 0.001 | 0.030 ± 0.000 | 0.039 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.052 ± 0.001 | 0.032 ± 0.000 | 0.041 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.093 ± 0.003 | 0.021 ± 0.002 | 0.147 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.089 ± 0.000 | 0.021 ± 0.000 | 0.158 ± 0.011 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.095 ± 0.002 | 0.022 ± 0.000 | 0.185 ± 0.014 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.057 ± 0.001 | 0.021 ± 0.000 | 0.073 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.054 ± 0.000 | 0.021 ± 0.000 | 0.076 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.057 ± 0.001 | 0.022 ± 0.000 | 0.090 ± 0.006 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.044 ± 0.001 | 0.017 ± 0.000 | 0.029 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.044 ± 0.001 | 0.018 ± 0.001 | 0.032 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.051 ± 0.001 | 0.027 ± 0.000 | 0.041 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.103 ± 0.001 | 0.016 ± 0.000 | 0.092 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.102 ± 0.000 | 0.018 ± 0.000 | 0.093 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.125 ± 0.001 | 0.027 ± 0.000 | 0.113 ± 0.003 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.050 ± 0.001 | 0.015 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.050 ± 0.001 | 0.017 ± 0.000 | 0.061 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.063 ± 0.000 | 0.026 ± 0.000 | 0.080 ± 0.008 | 0.018 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.017 ± 0.001 | 0.021 ± 0.001 | 0.006 ± 0.001 | 0.024 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.028 ± 0.002 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.030 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.021 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.011 ± 0.001 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 25.1x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-eager`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 58.4x faster than the slowest successful cell (`tenferro-eager`, 0.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 0.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 43.2x faster than the slowest successful cell (`tenferro-eager`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 71.0x faster than the slowest successful cell (`tenferro-eager`, 0.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.382 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 39.0x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 65.1x faster than the slowest successful cell (`tenferro-eager`, 0.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 59.6x faster than the slowest successful cell (`tenferro-eager`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 101.0x faster than the slowest successful cell (`tenferro-eager`, 0.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 27.7x faster than the slowest successful cell (`pytorch-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

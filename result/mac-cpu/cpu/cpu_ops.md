# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`
- Benchmark commit: `64970fd383409685da2dbf874a4ad9dcdff4fec0`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260916_162342/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260916_162342`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260916_162342`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_162342/cpu_ops_t1_20260916_162342.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_162342/cpu_ops_t1_20260916_162342.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 11.032 ± 0.051 | 11.054 ± 0.054 | 10.102 ± 0.090 | 10.575 ± 0.068 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.177 ± 0.002 | 0.181 ± 0.001 | 0.164 ± 0.004 | 0.163 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.774 ± 0.014 | 2.776 ± 0.028 | 2.602 ± 0.140 | 2.538 ± 0.038 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.698 ± 0.005 | 0.713 ± 0.014 | 0.630 ± 0.006 | 0.646 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.956 ± 0.011 | 0.958 ± 0.016 | 0.141 ± 0.005 | 0.260 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.001 | 0.021 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.253 ± 0.005 | 0.248 ± 0.003 | 0.037 ± 0.001 | 0.071 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.066 ± 0.000 | 0.066 ± 0.001 | 0.011 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.542 ± 0.018 | 1.633 ± 0.093 | 0.699 ± 0.008 | 0.879 ± 0.062 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.388 ± 0.005 | 0.402 ± 0.004 | 0.175 ± 0.003 | 0.208 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.001 | 0.104 ± 0.005 | 0.046 ± 0.001 | 0.056 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 3.492 ± 0.026 | 3.506 ± 0.025 | 2.658 ± 0.039 | 2.920 ± 0.068 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.060 ± 0.001 | 0.061 ± 0.001 | 0.043 ± 0.001 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.878 ± 0.009 | 0.902 ± 0.097 | 0.656 ± 0.053 | 0.687 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.223 ± 0.003 | 0.232 ± 0.006 | 0.165 ± 0.002 | 0.178 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.306 ± 0.019 | 0.305 ± 0.011 | 1.060 ± 0.063 | 0.400 ± 0.018 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.022 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.087 ± 0.003 | 0.089 ± 0.002 | 0.254 ± 0.021 | 0.068 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.030 ± 0.001 | 0.030 ± 0.002 | 0.069 ± 0.007 | 0.048 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.038 ± 0.000 | 0.040 ± 0.001 | 0.008 ± 0.001 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.040 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.052 ± 0.001 | 0.053 ± 0.002 | 0.020 ± 0.002 | 0.048 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.016 ± 0.000 | 0.018 ± 0.001 | 0.008 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.060 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.113 ± 0.004 | 0.116 ± 0.004 | 0.492 ± 0.014 | 0.081 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.035 ± 0.001 | 0.128 ± 0.004 | 0.046 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.036 ± 0.002 | 0.042 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 2.926 ± 0.070 | 2.911 ± 0.037 | 1.913 ± 0.064 | 2.044 ± 0.081 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.051 ± 0.001 | 0.053 ± 0.001 | 0.032 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.726 ± 0.011 | 0.736 ± 0.007 | 0.464 ± 0.042 | 0.498 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.191 ± 0.002 | 0.199 ± 0.003 | 0.119 ± 0.002 | 0.128 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.998 ± 0.010 | 1.012 ± 0.013 | 0.122 ± 0.003 | 0.161 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.022 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.269 ± 0.004 | 0.260 ± 0.004 | 0.032 ± 0.000 | 0.044 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.069 ± 0.001 | 0.070 ± 0.001 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.188 ± 0.019 | 1.278 ± 0.056 | 0.280 ± 0.006 | 0.325 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.302 ± 0.002 | 0.306 ± 0.004 | 0.071 ± 0.003 | 0.084 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.080 ± 0.001 | 0.082 ± 0.001 | 0.019 ± 0.000 | 0.025 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 1.655 ± 0.016 | 1.680 ± 0.018 | 0.693 ± 0.015 | 0.785 ± 0.055 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.032 ± 0.000 | 0.033 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.428 ± 0.002 | 0.435 ± 0.007 | 0.172 ± 0.007 | 0.188 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.110 ± 0.001 | 0.114 ± 0.001 | 0.045 ± 0.001 | 0.051 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.132 ± 0.026 | 2.673 ± 0.050 | 0.636 ± 0.012 | 0.817 ± 0.028 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.041 ± 0.001 | 0.061 ± 0.001 | 0.015 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.544 ± 0.009 | 0.692 ± 0.012 | 0.160 ± 0.005 | 0.177 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.143 ± 0.003 | 0.191 ± 0.003 | 0.044 ± 0.002 | 0.049 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.988 ± 0.009 | 1.979 ± 0.018 | 0.063 ± 0.004 | 0.078 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.022 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.256 ± 0.003 | 0.516 ± 0.006 | 0.020 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.068 ± 0.001 | 0.142 ± 0.002 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.053 ± 0.008 | 2.075 ± 0.046 | 0.100 ± 0.004 | 0.117 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.050 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.270 ± 0.003 | 0.531 ± 0.008 | 0.028 ± 0.001 | 0.034 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.072 ± 0.001 | 0.153 ± 0.003 | 0.011 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.285 ± 0.009 | 2.169 ± 0.026 | 0.220 ± 0.013 | 0.267 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.052 ± 0.001 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.332 ± 0.003 | 0.569 ± 0.004 | 0.058 ± 0.002 | 0.064 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.088 ± 0.001 | 0.158 ± 0.002 | 0.019 ± 0.000 | 0.021 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 21.765 ± 0.092 | 21.872 ± 0.078 | 20.514 ± 0.739 | 21.376 ± 0.202 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.350 ± 0.004 | 0.354 ± 0.006 | 0.323 ± 0.003 | 0.330 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.463 ± 0.051 | 5.492 ± 0.038 | 5.056 ± 0.027 | 5.143 ± 0.127 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.383 ± 0.009 | 1.427 ± 0.035 | 1.283 ± 0.010 | 1.313 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.592 ± 0.021 | 1.591 ± 0.013 | 0.325 ± 0.012 | 1.022 ± 0.037 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.031 ± 0.000 | 0.033 ± 0.001 | 0.008 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.431 ± 0.025 | 0.411 ± 0.005 | 0.083 ± 0.001 | 0.267 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.106 ± 0.001 | 0.107 ± 0.001 | 0.024 ± 0.001 | 0.069 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.684 ± 0.044 | 2.684 ± 0.026 | 1.412 ± 0.021 | 2.149 ± 0.114 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.048 ± 0.000 | 0.050 ± 0.001 | 0.024 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.679 ± 0.010 | 0.681 ± 0.008 | 0.369 ± 0.038 | 0.523 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.175 ± 0.001 | 0.176 ± 0.002 | 0.089 ± 0.001 | 0.136 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 6.553 ± 0.044 | 6.596 ± 0.094 | 5.306 ± 0.035 | 5.906 ± 0.126 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.108 ± 0.001 | 0.112 ± 0.002 | 0.085 ± 0.002 | 0.094 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.636 ± 0.009 | 1.676 ± 0.012 | 1.298 ± 0.019 | 1.426 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.418 ± 0.007 | 0.422 ± 0.005 | 0.331 ± 0.003 | 0.365 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.110 ± 0.027 | 1.011 ± 0.057 | 3.948 ± 0.196 | 0.949 ± 0.077 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.126 ± 0.002 | 0.043 ± 0.001 | 0.098 ± 0.003 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.344 ± 0.006 | 0.274 ± 0.006 | 0.967 ± 0.081 | 0.222 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.166 ± 0.003 | 0.097 ± 0.006 | 0.278 ± 0.009 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.192 ± 0.002 | 0.125 ± 0.001 | 0.042 ± 0.003 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.091 ± 0.001 | 0.022 ± 0.000 | 0.029 ± 0.000 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.122 ± 0.001 | 0.048 ± 0.001 | 0.032 ± 0.003 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.098 ± 0.001 | 0.027 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.245 ± 0.003 | 0.173 ± 0.004 | 0.076 ± 0.004 | 0.176 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.094 ± 0.001 | 0.023 ± 0.001 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.134 ± 0.002 | 0.060 ± 0.001 | 0.039 ± 0.001 | 0.048 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.002 | 0.033 ± 0.001 | 0.032 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.595 ± 0.016 | 0.569 ± 0.020 | 3.352 ± 0.127 | 0.240 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.111 ± 0.001 | 0.035 ± 0.000 | 0.086 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.223 ± 0.002 | 0.160 ± 0.004 | 0.861 ± 0.049 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.136 ± 0.001 | 0.063 ± 0.002 | 0.240 ± 0.003 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 9.127 ± 0.084 | 5.039 ± 0.036 | 1.001 ± 0.046 | 0.945 ± 0.034 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.260 ± 0.004 | 0.133 ± 0.001 | 0.037 ± 0.002 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.385 ± 0.007 | 1.304 ± 0.027 | 0.234 ± 0.013 | 0.208 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.677 ± 0.012 | 0.380 ± 0.005 | 0.077 ± 0.003 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.057 ± 0.049 | 3.658 ± 0.032 | 0.110 ± 0.007 | 0.106 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.179 ± 0.035 | 0.109 ± 0.002 | 0.023 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.106 ± 0.008 | 0.987 ± 0.012 | 0.043 ± 0.002 | 0.031 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.360 ± 0.005 | 0.281 ± 0.002 | 0.028 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.377 ± 0.060 | 3.818 ± 0.053 | 0.159 ± 0.007 | 0.159 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.180 ± 0.002 | 0.111 ± 0.002 | 0.024 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.182 ± 0.011 | 1.006 ± 0.015 | 0.055 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.385 ± 0.008 | 0.291 ± 0.002 | 0.031 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.393 ± 0.068 | 4.041 ± 0.028 | 0.333 ± 0.013 | 0.323 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.198 ± 0.003 | 0.115 ± 0.002 | 0.027 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.441 ± 0.014 | 1.080 ± 0.008 | 0.096 ± 0.005 | 0.078 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.447 ± 0.003 | 0.317 ± 0.008 | 0.041 ± 0.001 | 0.026 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.614 ± 0.005 | 0.632 ± 0.003 | 0.630 ± 0.008 | 0.669 ± 0.016 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.647 ± 0.046 | 2.676 ± 0.042 | 2.745 ± 0.031 | 2.906 ± 0.018 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.202 ± 0.001 | 0.206 ± 0.002 | 0.208 ± 0.008 | 0.214 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 75.552 ± 0.239 | 81.199 ± 0.603 | 139.174 ± 2.688 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 2.974 ± 0.041 | 3.016 ± 0.076 | 3.999 ± 0.023 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 13.923 ± 0.073 | 14.525 ± 0.127 | 21.997 ± 0.199 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 79.554 ± 0.993 | 69.656 ± 0.418 | 138.078 ± 11.930 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 2.953 ± 0.021 | 2.781 ± 0.025 | 3.960 ± 0.035 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 13.864 ± 0.099 | 12.892 ± 0.197 | 21.679 ± 0.220 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 31.801 ± 1.450 | 62.799 ± 19.106 | 87.654 ± 0.179 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.684 ± 0.006 | 1.316 ± 0.025 | 1.744 ± 0.020 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 4.185 ± 0.086 | 8.645 ± 0.118 | 12.279 ± 0.127 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 31.390 ± 1.140 | 32.842 ± 0.683 | 89.454 ± 0.867 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.704 ± 0.008 | 0.779 ± 0.019 | 1.747 ± 0.021 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 4.338 ± 0.107 | 4.740 ± 0.091 | 12.254 ± 0.173 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.033 ± 0.001 | 0.031 ± 0.001 | 0.019 ± 0.001 | 0.080 ± 0.002 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.014 ± 0.001 | 0.013 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.161 ± 0.003 | 0.093 ± 0.002 | 0.060 ± 0.002 | 0.227 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.117 ± 0.005 | 0.038 ± 0.000 | 0.024 ± 0.002 | 0.036 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 51.245 ± 0.493 | 57.627 ± 0.472 | 137.206 ± 2.573 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.880 ± 0.021 | 1.938 ± 0.024 | 3.195 ± 0.020 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 8.751 ± 0.060 | 10.072 ± 0.100 | 19.231 ± 0.179 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 51.363 ± 0.579 | 56.459 ± 6.403 | 144.051 ± 7.900 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.882 ± 0.022 | 1.852 ± 0.021 | 3.171 ± 0.021 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 8.985 ± 0.091 | 9.624 ± 0.154 | 19.966 ± 2.015 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.416 ± 0.003 | 0.134 ± 0.005 | 0.083 ± 0.003 | 0.064 ± 0.001 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.205 ± 0.004 | 0.076 ± 0.001 | 0.041 ± 0.004 | 0.022 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 6.032 ± 0.310 | 5.847 ± 0.082 | 5.591 ± 0.262 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.292 ± 0.003 | 0.293 ± 0.023 | 0.219 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.251 ± 0.045 | 1.203 ± 0.051 | 1.039 ± 0.028 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 6.030 ± 0.114 | 9.804 ± 0.124 | 5.274 ± 0.407 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.287 ± 0.004 | 0.388 ± 0.008 | 0.217 ± 0.005 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.197 ± 0.055 | 1.887 ± 0.085 | 0.996 ± 0.031 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 2.537 ± 0.028 | 1.271 ± 0.175 | 1.274 ± 0.022 | 1.387 ± 0.017 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.637 ± 0.017 | 0.304 ± 0.002 | 0.312 ± 0.018 | 0.310 ± 0.005 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 118.460 ± 0.235 | 137.047 ± 12.934 | 180.814 ± 2.313 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.065 ± 0.050 | 5.539 ± 0.029 | 6.154 ± 0.039 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 23.312 ± 0.133 | 26.189 ± 0.100 | 31.421 ± 0.299 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 117.921 ± 0.442 | 113.731 ± 0.414 | 179.795 ± 15.460 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 5.043 ± 0.056 | 5.006 ± 0.045 | 6.111 ± 0.041 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 23.293 ± 0.409 | 22.862 ± 0.122 | 33.517 ± 2.970 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 5.029 ± 0.267 | 5.510 ± 0.371 | 5.162 ± 1.558 | 33.119 ± 0.517 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.023 ± 0.001 | 0.023 ± 0.002 | 0.015 ± 0.001 | 0.071 ± 0.001 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.105 ± 0.004 | 0.107 ± 0.005 | 0.101 ± 0.005 | 0.552 ± 0.006 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.657 ± 0.020 | 0.669 ± 0.018 | 0.623 ± 0.060 | 4.354 ± 0.136 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.321 ± 0.030 | 1.325 ± 0.019 | 1.291 ± 0.142 | 8.284 ± 0.422 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.374 ± 0.020 | 0.378 ± 0.008 | 0.344 ± 0.024 | 2.163 ± 0.040 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.380 ± 0.003 | 0.384 ± 0.004 | 0.405 ± 0.004 | 0.409 ± 0.024 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.181 ± 0.006 | 1.185 ± 0.010 | 1.360 ± 0.018 | 1.344 ± 0.116 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.072 ± 0.000 | 0.074 ± 0.001 | 0.072 ± 0.002 | 0.073 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.076 ± 0.001 | 0.080 ± 0.001 | 0.052 ± 0.001 | 0.056 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.079 ± 0.004 | 0.081 ± 0.006 | 0.052 ± 0.010 | 0.100 ± 0.027 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.101 ± 0.002 | 0.096 ± 0.002 | 0.065 ± 0.009 | 0.118 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.328 ± 0.031 | 0.213 ± 0.005 | 0.201 ± 0.005 | 0.217 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.358 ± 0.042 | 0.216 ± 0.003 | 0.201 ± 0.013 | 0.285 ± 0.019 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.386 ± 0.017 | 0.269 ± 0.003 | 0.235 ± 0.010 | 0.325 ± 0.015 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.034 ± 0.000 | 0.018 ± 0.001 | 0.018 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.031 ± 0.001 | 0.037 ± 0.001 | 0.019 ± 0.002 | 0.029 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.038 ± 0.000 | 0.045 ± 0.002 | 0.029 ± 0.010 | 0.055 ± 0.011 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.222 ± 0.010 | 1.235 ± 0.007 | 1.228 ± 0.012 | 1.278 ± 0.034 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.868 ± 0.023 | 4.894 ± 0.033 | 4.907 ± 0.029 | 5.037 ± 0.053 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.279 ± 0.002 | 0.280 ± 0.002 | 0.289 ± 0.012 | 0.281 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.042 ± 0.000 | 0.044 ± 0.000 | 0.040 ± 0.001 | 0.040 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.038 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.042 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.042 ± 0.000 | 0.051 ± 0.002 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.026 ± 0.000 | 0.034 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.073 ± 0.001 | 0.077 ± 0.001 | 0.045 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.028 ± 0.007 | 0.035 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.003 | 0.043 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.041 ± 0.001 | 0.045 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.025 ± 0.000 | 0.029 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.066 ± 0.000 | 0.071 ± 0.002 | 0.046 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.025 ± 0.000 | 0.030 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.001 | 0.037 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.000 | 0.062 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.000 | 0.051 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.050 ± 0.000 | 0.075 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.031 ± 0.000 | 0.051 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.001 | 0.058 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.001 | 0.048 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.000 | 0.041 ± 0.013 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.054 ± 0.009 | 0.057 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.031 ± 0.000 | 0.040 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.038 ± 0.001 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | 0.101 ± 0.001 | 0.026 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.085 ± 0.001 | 0.019 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | 0.105 ± 0.001 | 0.029 ± 0.000 | 0.018 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.086 ± 0.002 | 0.019 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.097 ± 0.001 | 0.024 ± 0.000 | 0.015 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.064 ± 0.000 | 0.047 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.048 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.075 ± 0.001 | 0.056 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.048 ± 0.001 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.049 ± 0.000 | 0.045 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.063 ± 0.001 | 0.047 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.049 ± 0.001 | 0.038 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.075 ± 0.000 | 0.055 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.049 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.056 ± 0.001 | 0.044 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | 0.119 ± 0.002 | 0.046 ± 0.001 | 0.021 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.114 ± 0.001 | 0.045 ± 0.001 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | 0.151 ± 0.001 | 0.058 ± 0.001 | 0.028 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.115 ± 0.003 | 0.045 ± 0.001 | 0.019 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.117 ± 0.001 | 0.045 ± 0.001 | 0.020 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.036 ± 0.000 | 0.085 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.034 ± 0.000 | 0.086 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.046 ± 0.002 | 0.093 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.034 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.035 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.035 ± 0.001 | 0.039 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.044 ± 0.000 | 0.050 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.033 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.033 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | 0.123 ± 0.001 | 0.042 ± 0.001 | 0.040 ± 0.002 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.093 ± 0.001 | 0.016 ± 0.000 | 0.014 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | 0.204 ± 0.002 | 0.085 ± 0.002 | 0.085 ± 0.002 | 0.077 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.078 ± 0.002 | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.087 ± 0.001 | 0.021 ± 0.001 | 0.020 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.044 ± 0.001 | 0.076 ± 0.003 | 0.026 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.086 ± 0.002 | 0.125 ± 0.004 | 0.074 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.019 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.028 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.043 ± 0.000 | 0.057 ± 0.001 | 0.027 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.016 ± 0.000 | 0.031 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.085 ± 0.002 | 0.103 ± 0.003 | 0.075 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.017 ± 0.000 | 0.032 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.022 ± 0.000 | 0.036 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.013 ± 0.000 | 0.023 ± 0.001 | 0.009 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.013 ± 0.000 | 0.023 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.022 ± 0.000 | 0.025 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.073 ± 0.000 | 0.075 ± 0.000 | 0.070 ± 0.001 | 0.073 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 1.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 31.4x faster than the slowest successful cell (`tenferro-trace`, 1.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`tenferro-trace`, 0.516 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 0.516 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 17.1x faster than the slowest successful cell (`tenferro-trace`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-trace`, 2.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 2.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-trace`, 0.531 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 19.2x faster than the slowest successful cell (`tenferro-trace`, 0.531 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-trace`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-eager`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 3.352 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.861 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.240 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 2.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-eager`, 2.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`16x16xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 38.4x faster than the slowest successful cell (`tenferro-eager`, 4.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 36.9x faster than the slowest successful cell (`tenferro-eager`, 4.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 36.0x faster than the slowest successful cell (`tenferro-eager`, 1.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 25.6x faster than the slowest successful cell (`tenferro-eager`, 1.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 37.5x faster than the slowest successful cell (`tenferro-eager`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 27.6x faster than the slowest successful cell (`tenferro-eager`, 4.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 27.6x faster than the slowest successful cell (`tenferro-eager`, 4.377 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 30.1x faster than the slowest successful cell (`tenferro-eager`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 1.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 1.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 28.1x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 5.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 16.2x faster than the slowest successful cell (`tenferro-eager`, 5.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 1.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 1.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`16x16`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.3x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`32x32`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-trace`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-trace`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.4x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 20.4x faster than the slowest successful cell (`tenferro-eager`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.6x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 0.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260916_164245/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260916_164245`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260916_164245`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260916_164245/cpu_ops_t4_20260916_164245.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260916_164245/cpu_ops_t4_20260916_164245.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 11.082 ± 0.080 | 10.924 ± 0.086 | 9.850 ± 0.087 | 5.364 ± 2.538 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.177 ± 0.002 | 0.180 ± 0.002 | 0.158 ± 0.002 | 0.141 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.868 ± 0.049 | 2.773 ± 0.028 | 2.471 ± 0.037 | 1.169 ± 0.096 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.693 ± 0.004 | 0.696 ± 0.005 | 0.629 ± 0.005 | 0.251 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.969 ± 0.009 | 0.952 ± 0.009 | 0.132 ± 0.006 | 0.269 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.244 ± 0.002 | 0.244 ± 0.002 | 0.034 ± 0.001 | 0.072 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.065 ± 0.001 | 0.066 ± 0.001 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.556 ± 0.010 | 1.527 ± 0.025 | 0.691 ± 0.009 | 0.558 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.396 ± 0.003 | 0.390 ± 0.003 | 0.174 ± 0.004 | 0.211 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.000 | 0.102 ± 0.001 | 0.045 ± 0.001 | 0.057 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.540 ± 0.043 | 3.510 ± 0.047 | 2.542 ± 0.045 | 1.257 ± 0.188 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.059 ± 0.001 | 0.060 ± 0.001 | 0.042 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.902 ± 0.023 | 0.874 ± 0.008 | 0.649 ± 0.009 | 0.324 ± 0.032 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.222 ± 0.002 | 0.221 ± 0.002 | 0.165 ± 0.003 | 0.177 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.323 ± 0.010 | 0.286 ± 0.014 | 1.010 ± 0.028 | 0.295 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.095 ± 0.002 | 0.101 ± 0.004 | 0.249 ± 0.004 | 0.101 ± 0.018 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.028 ± 0.001 | 0.026 ± 0.001 | 0.074 ± 0.008 | 0.048 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.038 ± 0.001 | 0.039 ± 0.000 | 0.007 ± 0.000 | 0.044 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.005 ± 0.001 | 0.040 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.003 | 0.052 ± 0.001 | 0.027 ± 0.003 | 0.093 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.017 ± 0.001 | 0.018 ± 0.000 | 0.008 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.123 ± 0.001 | 0.122 ± 0.002 | 0.479 ± 0.016 | 0.080 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.033 ± 0.000 | 0.127 ± 0.004 | 0.046 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.036 ± 0.001 | 0.045 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.869 ± 0.023 | 2.861 ± 0.033 | 1.710 ± 0.024 | 1.133 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.051 ± 0.000 | 0.053 ± 0.000 | 0.044 ± 0.002 | 0.037 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.787 ± 0.027 | 0.766 ± 0.029 | 0.463 ± 0.008 | 0.292 ± 0.023 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.188 ± 0.001 | 0.192 ± 0.002 | 0.128 ± 0.004 | 0.161 ± 0.029 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.016 ± 0.007 | 0.993 ± 0.009 | 0.120 ± 0.002 | 0.163 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.258 ± 0.002 | 0.254 ± 0.002 | 0.043 ± 0.001 | 0.045 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.067 ± 0.000 | 0.069 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.191 ± 0.007 | 1.188 ± 0.009 | 0.269 ± 0.003 | 0.334 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.306 ± 0.002 | 0.299 ± 0.002 | 0.082 ± 0.002 | 0.086 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.079 ± 0.001 | 0.081 ± 0.001 | 0.033 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 1.676 ± 0.017 | 1.656 ± 0.032 | 0.671 ± 0.012 | 0.381 ± 0.023 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.031 ± 0.000 | 0.033 ± 0.000 | 0.025 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.428 ± 0.006 | 0.427 ± 0.009 | 0.177 ± 0.003 | 0.200 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.109 ± 0.001 | 0.110 ± 0.002 | 0.057 ± 0.001 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.172 ± 0.030 | 2.563 ± 0.040 | 0.280 ± 0.007 | 0.569 ± 0.057 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.041 ± 0.001 | 0.060 ± 0.001 | 0.026 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.571 ± 0.017 | 0.694 ± 0.008 | 0.106 ± 0.006 | 0.148 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.142 ± 0.002 | 0.186 ± 0.001 | 0.041 ± 0.002 | 0.078 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.998 ± 0.022 | 1.962 ± 0.027 | 0.060 ± 0.004 | 0.080 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.022 ± 0.000 | 0.048 ± 0.001 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.253 ± 0.002 | 0.503 ± 0.004 | 0.031 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.068 ± 0.000 | 0.139 ± 0.001 | 0.021 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 1.192 ± 0.060 | 2.001 ± 0.015 | 0.071 ± 0.003 | 0.122 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.049 ± 0.001 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.270 ± 0.001 | 0.517 ± 0.003 | 0.034 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.073 ± 0.000 | 0.143 ± 0.001 | 0.023 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.310 ± 0.008 | 2.173 ± 0.027 | 0.130 ± 0.005 | 0.176 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.052 ± 0.000 | 0.021 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.339 ± 0.008 | 0.553 ± 0.006 | 0.049 ± 0.002 | 0.068 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.087 ± 0.000 | 0.181 ± 0.005 | 0.028 ± 0.002 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 21.872 ± 0.129 | 21.760 ± 0.334 | 20.022 ± 0.254 | 10.868 ± 0.218 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.346 ± 0.002 | 0.349 ± 0.004 | 0.318 ± 0.003 | 0.180 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.615 ± 0.062 | 6.159 ± 0.300 | 5.027 ± 0.108 | 2.314 ± 0.577 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.370 ± 0.013 | 1.381 ± 0.016 | 1.279 ± 0.011 | 0.442 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.570 ± 0.019 | 1.558 ± 0.026 | 0.320 ± 0.005 | 1.058 ± 0.055 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.030 ± 0.000 | 0.032 ± 0.001 | 0.007 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.399 ± 0.003 | 0.396 ± 0.004 | 0.084 ± 0.003 | 0.269 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.104 ± 0.001 | 0.107 ± 0.001 | 0.022 ± 0.001 | 0.071 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.697 ± 0.048 | 2.648 ± 0.042 | 1.382 ± 0.035 | 0.685 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.048 ± 0.000 | 0.050 ± 0.001 | 0.024 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.680 ± 0.006 | 0.666 ± 0.003 | 0.343 ± 0.009 | 0.515 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.174 ± 0.001 | 0.173 ± 0.002 | 0.089 ± 0.001 | 0.135 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.589 ± 0.370 | 6.517 ± 0.060 | 5.130 ± 0.172 | 2.410 ± 0.405 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.108 ± 0.001 | 0.111 ± 0.002 | 0.084 ± 0.001 | 0.097 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.670 ± 0.026 | 1.633 ± 0.030 | 1.288 ± 0.016 | 0.487 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.414 ± 0.003 | 0.412 ± 0.004 | 0.331 ± 0.006 | 0.303 ± 0.029 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.058 ± 0.020 | 0.960 ± 0.046 | 3.884 ± 0.130 | 0.620 ± 0.041 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.127 ± 0.001 | 0.044 ± 0.001 | 0.094 ± 0.003 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.412 ± 0.013 | 0.323 ± 0.006 | 1.000 ± 0.026 | 0.206 ± 0.020 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.173 ± 0.001 | 0.106 ± 0.002 | 0.282 ± 0.010 | 0.061 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.192 ± 0.004 | 0.126 ± 0.002 | 0.041 ± 0.002 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.093 ± 0.001 | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.117 ± 0.001 | 0.047 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.097 ± 0.002 | 0.027 ± 0.000 | 0.029 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.263 ± 0.019 | 0.175 ± 0.005 | 0.093 ± 0.004 | 0.266 ± 0.020 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.092 ± 0.001 | 0.023 ± 0.000 | 0.029 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.136 ± 0.005 | 0.061 ± 0.001 | 0.041 ± 0.001 | 0.070 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.001 | 0.030 ± 0.000 | 0.033 ± 0.001 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.668 ± 0.008 | 0.609 ± 0.013 | 3.085 ± 0.025 | 0.205 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.110 ± 0.001 | 0.034 ± 0.001 | 0.083 ± 0.002 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.238 ± 0.002 | 0.170 ± 0.002 | 0.820 ± 0.038 | 0.063 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.137 ± 0.002 | 0.063 ± 0.002 | 0.238 ± 0.007 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 9.141 ± 0.141 | 4.906 ± 0.084 | 0.595 ± 0.167 | 0.712 ± 0.022 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.255 ± 0.003 | 0.131 ± 0.002 | 0.047 ± 0.004 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 2.474 ± 0.033 | 1.329 ± 0.016 | 0.214 ± 0.017 | 0.194 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.683 ± 0.008 | 0.364 ± 0.004 | 0.083 ± 0.010 | 0.085 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 4.102 ± 0.042 | 3.640 ± 0.019 | 0.103 ± 0.004 | 0.108 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.175 ± 0.002 | 0.107 ± 0.002 | 0.037 ± 0.001 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 1.096 ± 0.006 | 0.954 ± 0.013 | 0.056 ± 0.003 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.365 ± 0.009 | 0.278 ± 0.002 | 0.044 ± 0.003 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 4.480 ± 0.079 | 3.769 ± 0.029 | 0.131 ± 0.003 | 0.162 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.178 ± 0.002 | 0.110 ± 0.001 | 0.039 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 1.212 ± 0.028 | 0.983 ± 0.008 | 0.064 ± 0.003 | 0.045 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.375 ± 0.003 | 0.285 ± 0.002 | 0.047 ± 0.003 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 5.459 ± 0.063 | 3.948 ± 0.047 | 0.246 ± 0.020 | 0.238 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.195 ± 0.002 | 0.115 ± 0.001 | 0.044 ± 0.003 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 1.460 ± 0.013 | 1.050 ± 0.012 | 0.090 ± 0.008 | 0.087 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.449 ± 0.013 | 0.350 ± 0.020 | 0.056 ± 0.003 | 0.028 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.597 ± 0.008 | 0.626 ± 0.002 | 0.628 ± 0.019 | 0.667 ± 0.008 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.527 ± 0.016 | 2.658 ± 0.023 | 2.690 ± 0.046 | 2.779 ± 0.026 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.201 ± 0.002 | 0.205 ± 0.001 | 0.203 ± 0.002 | 0.215 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 73.377 ± 0.873 | 79.092 ± 0.458 | 84.400 ± 1.222 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 2.966 ± 0.025 | 2.970 ± 0.023 | 3.066 ± 0.058 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 13.280 ± 0.090 | 14.135 ± 0.165 | 15.106 ± 0.110 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 73.753 ± 0.173 | 68.030 ± 0.238 | 85.039 ± 3.362 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 2.937 ± 0.016 | 2.720 ± 0.027 | 3.073 ± 0.045 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 13.705 ± 0.354 | 12.528 ± 0.072 | 15.075 ± 0.062 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 27.751 ± 0.376 | 54.281 ± 1.146 | 35.211 ± 0.636 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.762 ± 0.008 | 1.341 ± 0.015 | 0.760 ± 0.009 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.191 ± 0.044 | 7.565 ± 0.103 | 5.442 ± 0.179 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 28.158 ± 0.395 | 29.437 ± 0.215 | 36.130 ± 0.330 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.739 ± 0.010 | 0.851 ± 0.006 | 0.809 ± 0.014 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.161 ± 0.020 | 4.193 ± 0.023 | 5.453 ± 0.077 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.037 ± 0.001 | 0.038 ± 0.000 | 0.018 ± 0.001 | 0.079 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.016 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.163 ± 0.002 | 0.099 ± 0.002 | 0.059 ± 0.002 | 0.186 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.115 ± 0.010 | 0.040 ± 0.001 | 0.024 ± 0.002 | 0.037 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 47.134 ± 0.938 | 51.033 ± 0.736 | 58.636 ± 1.404 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 1.937 ± 0.018 | 1.860 ± 0.029 | 1.821 ± 0.068 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 8.467 ± 0.328 | 8.362 ± 0.065 | 9.985 ± 0.654 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 47.300 ± 2.675 | 50.427 ± 1.784 | 63.491 ± 0.467 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 1.906 ± 0.027 | 1.783 ± 0.018 | 1.880 ± 0.015 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 8.253 ± 0.140 | 8.119 ± 0.054 | 10.273 ± 0.078 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.420 ± 0.008 | 0.135 ± 0.003 | 0.078 ± 0.003 | 0.062 ± 0.004 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.203 ± 0.004 | 0.078 ± 0.001 | 0.039 ± 0.001 | 0.022 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.375 ± 0.042 | 5.160 ± 0.065 | 5.042 ± 0.100 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.326 ± 0.004 | 0.291 ± 0.009 | 0.219 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.209 ± 0.072 | 1.033 ± 0.009 | 1.100 ± 0.052 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.307 ± 0.285 | 8.758 ± 0.046 | 4.839 ± 0.259 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.317 ± 0.005 | 0.407 ± 0.003 | 0.212 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.167 ± 0.009 | 1.718 ± 0.018 | 1.025 ± 0.073 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 2.529 ± 0.013 | 1.257 ± 0.005 | 1.254 ± 0.029 | 1.396 ± 0.013 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.627 ± 0.010 | 0.303 ± 0.001 | 0.300 ± 0.005 | 0.308 ± 0.003 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 116.890 ± 2.352 | 131.926 ± 0.546 | 128.462 ± 0.898 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.023 ± 0.033 | 5.574 ± 0.096 | 5.289 ± 0.031 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 23.422 ± 0.572 | 25.003 ± 0.181 | 24.338 ± 0.094 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 115.795 ± 0.397 | 111.717 ± 0.399 | 127.481 ± 1.105 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 4.980 ± 0.018 | 4.937 ± 0.045 | 5.226 ± 0.031 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 23.374 ± 0.252 | 22.390 ± 0.043 | 24.206 ± 0.073 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.921 ± 0.329 | 4.795 ± 0.301 | 5.581 ± 1.317 | 8.669 ± 0.245 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.001 | 0.073 ± 0.001 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.119 ± 0.003 | 0.118 ± 0.003 | 0.099 ± 0.004 | 0.171 ± 0.008 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.653 ± 0.017 | 0.653 ± 0.008 | 0.607 ± 0.013 | 1.175 ± 0.040 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.292 ± 0.008 | 1.301 ± 0.018 | 1.295 ± 0.066 | 2.169 ± 0.067 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.351 ± 0.009 | 0.354 ± 0.013 | 0.344 ± 0.023 | 0.644 ± 0.031 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.374 ± 0.003 | 0.377 ± 0.002 | 0.396 ± 0.007 | 0.396 ± 0.006 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.175 ± 0.006 | 1.181 ± 0.007 | 1.254 ± 0.077 | 1.231 ± 0.014 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.072 ± 0.000 | 0.073 ± 0.000 | 0.083 ± 0.001 | 0.073 ± 0.000 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.075 ± 0.002 | 0.071 ± 0.001 | 0.051 ± 0.002 | 0.057 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.079 ± 0.002 | 0.073 ± 0.001 | 0.050 ± 0.005 | 0.081 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.101 ± 0.000 | 0.089 ± 0.003 | 0.063 ± 0.004 | 0.098 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.299 ± 0.011 | 0.241 ± 0.001 | 0.188 ± 0.011 | 0.200 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.307 ± 0.014 | 0.243 ± 0.005 | 0.185 ± 0.020 | 0.222 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.355 ± 0.008 | 0.294 ± 0.008 | 0.215 ± 0.007 | 0.265 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.027 ± 0.001 | 0.034 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.000 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.029 ± 0.000 | 0.036 ± 0.001 | 0.019 ± 0.003 | 0.025 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.038 ± 0.000 | 0.043 ± 0.002 | 0.024 ± 0.001 | 0.049 ± 0.008 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.229 ± 0.022 | 1.216 ± 0.005 | 1.228 ± 0.024 | 1.256 ± 0.019 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.816 ± 0.047 | 4.819 ± 0.015 | 4.888 ± 0.041 | 4.868 ± 0.083 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.279 ± 0.002 | 0.280 ± 0.003 | 0.279 ± 0.004 | 0.287 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.042 ± 0.001 | 0.044 ± 0.000 | 0.039 ± 0.001 | 0.041 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.042 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.041 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.042 ± 0.001 | 0.050 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.000 | 0.033 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.072 ± 0.001 | 0.073 ± 0.002 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.026 ± 0.000 | 0.034 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.033 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.038 ± 0.004 | 0.042 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.066 ± 0.000 | 0.066 ± 0.001 | 0.044 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.040 ± 0.001 | 0.142 ± 0.006 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.031 ± 0.000 | 0.128 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.052 ± 0.001 | 0.148 ± 0.009 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.031 ± 0.000 | 0.129 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.031 ± 0.000 | 0.132 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.000 | 0.104 ± 0.010 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.031 ± 0.000 | 0.094 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.054 ± 0.008 | 0.107 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.031 ± 0.001 | 0.095 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.037 ± 0.000 | 0.099 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | 0.101 ± 0.002 | 0.026 ± 0.000 | 0.016 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.086 ± 0.001 | 0.019 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | 0.104 ± 0.003 | 0.028 ± 0.001 | 0.017 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.085 ± 0.003 | 0.019 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.097 ± 0.002 | 0.024 ± 0.002 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.063 ± 0.001 | 0.075 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.049 ± 0.001 | 0.068 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.076 ± 0.000 | 0.082 ± 0.006 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.048 ± 0.000 | 0.071 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.049 ± 0.001 | 0.073 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.062 ± 0.001 | 0.075 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.049 ± 0.001 | 0.074 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.075 ± 0.000 | 0.086 ± 0.017 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.049 ± 0.000 | 0.073 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.056 ± 0.001 | 0.075 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | 0.120 ± 0.001 | 0.046 ± 0.000 | 0.020 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.115 ± 0.003 | 0.045 ± 0.001 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | 0.149 ± 0.001 | 0.058 ± 0.002 | 0.027 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.119 ± 0.003 | 0.044 ± 0.000 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.118 ± 0.002 | 0.045 ± 0.000 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.036 ± 0.001 | 0.086 ± 0.006 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.034 ± 0.000 | 0.081 ± 0.002 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.048 ± 0.001 | 0.091 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.034 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.035 ± 0.000 | 0.082 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.034 ± 0.001 | 0.040 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.043 ± 0.002 | 0.051 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.033 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.034 ± 0.001 | 0.038 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | 0.122 ± 0.000 | 0.042 ± 0.000 | 0.039 ± 0.001 | 0.028 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.092 ± 0.001 | 0.016 ± 0.000 | 0.014 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | 0.204 ± 0.002 | 0.087 ± 0.001 | 0.084 ± 0.003 | 0.074 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.079 ± 0.002 | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.087 ± 0.001 | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.043 ± 0.000 | 0.078 ± 0.003 | 0.026 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.018 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.088 ± 0.002 | 0.122 ± 0.003 | 0.073 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.019 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.028 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.042 ± 0.000 | 0.057 ± 0.002 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.016 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.087 ± 0.001 | 0.098 ± 0.004 | 0.075 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.017 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.021 ± 0.000 | 0.036 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.013 ± 0.000 | 0.024 ± 0.001 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.012 ± 0.000 | 0.024 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.022 ± 0.000 | 0.026 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.073 ± 0.000 | 0.075 ± 0.000 | 0.070 ± 0.001 | 0.073 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 24.6x faster than the slowest successful cell (`tenferro-trace`, 1.962 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 32.9x faster than the slowest successful cell (`tenferro-trace`, 1.962 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 21.4x faster than the slowest successful cell (`tenferro-trace`, 0.503 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 0.503 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 21.2x faster than the slowest successful cell (`tenferro-trace`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 2.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 28.1x faster than the slowest successful cell (`tenferro-trace`, 2.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 0.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 15.3x faster than the slowest successful cell (`tenferro-trace`, 0.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 2.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 16.7x faster than the slowest successful cell (`tenferro-trace`, 2.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 22.8x faster than the slowest successful cell (`tenferro-eager`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 3.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.238 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 9.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 9.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 2.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 2.474 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 38.0x faster than the slowest successful cell (`tenferro-eager`, 4.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 39.7x faster than the slowest successful cell (`tenferro-eager`, 4.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 26.3x faster than the slowest successful cell (`tenferro-eager`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 33.8x faster than the slowest successful cell (`tenferro-eager`, 1.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 1.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 38.0x faster than the slowest successful cell (`tenferro-eager`, 0.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 27.7x faster than the slowest successful cell (`tenferro-eager`, 4.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 34.2x faster than the slowest successful cell (`tenferro-eager`, 4.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 30.6x faster than the slowest successful cell (`tenferro-eager`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 1.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 1.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-eager`, 5.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 5.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 0.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 1.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 1.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.449 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 34.5x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 25.4x faster than the slowest successful cell (`pytorch-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`16x16`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`32x32`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.2x faster than the slowest successful cell (`tenferro-eager`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 26.0x faster than the slowest successful cell (`tenferro-eager`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 23.1x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`tenferro-trace`, 0.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 0.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 0.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

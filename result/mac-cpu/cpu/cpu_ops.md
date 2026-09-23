# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`
- Benchmark commit: `5df994b8aee8e926b52e885ab89a5732e4e2ff8f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_201832/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_201832`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_201832`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_201832/cpu_ops_t1_20260923_201832.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_201832/cpu_ops_t1_20260923_201832.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 9.873 ± 0.053 | 9.830 ± 0.032 | 9.833 ± 0.141 | 10.261 ± 0.091 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.163 ± 0.002 | 0.160 ± 0.005 | 0.157 ± 0.004 | 0.157 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.465 ± 0.016 | 2.473 ± 0.034 | 2.468 ± 0.020 | 2.532 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.618 ± 0.003 | 0.621 ± 0.013 | 0.621 ± 0.006 | 0.630 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.122 ± 0.001 | 0.121 ± 0.001 | 0.129 ± 0.002 | 0.265 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.034 ± 0.001 | 0.034 ± 0.001 | 0.070 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.665 ± 0.012 | 0.674 ± 0.016 | 0.676 ± 0.014 | 0.809 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.013 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.169 ± 0.002 | 0.171 ± 0.003 | 0.171 ± 0.003 | 0.204 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.001 | 0.046 ± 0.001 | 0.045 ± 0.001 | 0.055 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 2.570 ± 0.038 | 2.563 ± 0.056 | 2.579 ± 0.061 | 2.722 ± 0.035 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.043 ± 0.001 | 0.044 ± 0.001 | 0.042 ± 0.001 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.641 ± 0.014 | 0.640 ± 0.011 | 0.640 ± 0.015 | 0.680 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.162 ± 0.004 | 0.164 ± 0.004 | 0.162 ± 0.003 | 0.170 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.243 ± 0.026 | 0.302 ± 0.002 | 1.000 ± 0.053 | 0.348 ± 0.027 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.068 ± 0.002 | 0.082 ± 0.000 | 0.250 ± 0.009 | 0.073 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.028 ± 0.001 | 0.065 ± 0.003 | 0.049 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.001 | 0.033 ± 0.001 | 0.008 ± 0.000 | 0.043 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.038 ± 0.002 | 0.045 ± 0.001 | 0.018 ± 0.001 | 0.047 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.011 ± 0.001 | 0.015 ± 0.001 | 0.008 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.085 ± 0.003 | 0.106 ± 0.004 | 0.480 ± 0.017 | 0.068 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.023 ± 0.000 | 0.030 ± 0.000 | 0.125 ± 0.003 | 0.047 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.034 ± 0.002 | 0.042 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.824 ± 0.022 | 1.835 ± 0.046 | 1.846 ± 0.023 | 1.945 ± 0.057 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.033 ± 0.001 | 0.034 ± 0.001 | 0.030 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.462 ± 0.010 | 0.474 ± 0.005 | 0.455 ± 0.018 | 0.486 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.121 ± 0.003 | 0.120 ± 0.002 | 0.116 ± 0.002 | 0.124 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.124 ± 0.002 | 0.125 ± 0.002 | 0.118 ± 0.001 | 0.158 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.034 ± 0.000 | 0.035 ± 0.001 | 0.031 ± 0.000 | 0.043 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.270 ± 0.003 | 0.276 ± 0.003 | 0.272 ± 0.007 | 0.318 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.070 ± 0.001 | 0.072 ± 0.001 | 0.070 ± 0.001 | 0.082 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.672 ± 0.013 | 0.676 ± 0.011 | 0.677 ± 0.024 | 0.732 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.174 ± 0.004 | 0.173 ± 0.004 | 0.171 ± 0.004 | 0.184 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.045 ± 0.001 | 0.047 ± 0.000 | 0.044 ± 0.001 | 0.051 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.588 ± 0.017 | 0.580 ± 0.032 | 0.602 ± 0.040 | 0.740 ± 0.032 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.014 ± 0.001 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.150 ± 0.006 | 0.149 ± 0.003 | 0.153 ± 0.004 | 0.175 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.041 ± 0.002 | 0.043 ± 0.001 | 0.042 ± 0.002 | 0.047 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.055 ± 0.001 | 0.055 ± 0.002 | 0.059 ± 0.001 | 0.075 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.088 ± 0.004 | 0.094 ± 0.002 | 0.096 ± 0.004 | 0.110 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.025 ± 0.001 | 0.028 ± 0.001 | 0.028 ± 0.001 | 0.033 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.196 ± 0.009 | 0.198 ± 0.005 | 0.212 ± 0.008 | 0.242 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.055 ± 0.001 | 0.054 ± 0.003 | 0.057 ± 0.001 | 0.062 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.020 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 19.923 ± 0.183 | 19.916 ± 0.105 | 20.010 ± 0.116 | 20.646 ± 0.124 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.317 ± 0.005 | 0.320 ± 0.008 | 0.318 ± 0.005 | 0.321 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.001 ± 0.190 | 5.070 ± 0.074 | 4.973 ± 0.065 | 5.102 ± 0.065 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.269 ± 0.030 | 1.250 ± 0.016 | 1.250 ± 0.028 | 1.287 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.308 ± 0.004 | 0.310 ± 0.004 | 0.314 ± 0.006 | 1.036 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.081 ± 0.002 | 0.083 ± 0.001 | 0.079 ± 0.001 | 0.247 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.024 ± 0.000 | 0.025 ± 0.000 | 0.022 ± 0.001 | 0.069 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.341 ± 0.020 | 1.359 ± 0.023 | 1.373 ± 0.028 | 1.993 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.025 ± 0.001 | 0.027 ± 0.000 | 0.024 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.341 ± 0.007 | 0.346 ± 0.007 | 0.341 ± 0.007 | 0.501 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.089 ± 0.002 | 0.091 ± 0.002 | 0.087 ± 0.002 | 0.130 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 5.129 ± 0.183 | 5.100 ± 0.085 | 5.094 ± 0.098 | 5.649 ± 0.251 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.084 ± 0.002 | 0.087 ± 0.002 | 0.083 ± 0.002 | 0.092 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.270 ± 0.012 | 1.279 ± 0.023 | 1.283 ± 0.028 | 1.411 ± 0.023 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.320 ± 0.008 | 0.332 ± 0.006 | 0.322 ± 0.011 | 0.354 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.984 ± 0.063 | 0.932 ± 0.036 | 3.860 ± 0.064 | 0.882 ± 0.037 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.108 ± 0.001 | 0.038 ± 0.000 | 0.095 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.296 ± 0.006 | 0.255 ± 0.005 | 0.962 ± 0.020 | 0.221 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.137 ± 0.002 | 0.092 ± 0.001 | 0.269 ± 0.004 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.155 ± 0.003 | 0.105 ± 0.001 | 0.039 ± 0.001 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.082 ± 0.002 | 0.017 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.103 ± 0.002 | 0.038 ± 0.000 | 0.031 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.085 ± 0.002 | 0.021 ± 0.000 | 0.029 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.199 ± 0.004 | 0.148 ± 0.003 | 0.073 ± 0.003 | 0.174 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.002 | 0.018 ± 0.000 | 0.028 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.113 ± 0.001 | 0.050 ± 0.001 | 0.040 ± 0.001 | 0.049 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.087 ± 0.002 | 0.024 ± 0.000 | 0.031 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.539 ± 0.007 | 0.527 ± 0.014 | 3.189 ± 0.063 | 0.219 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.092 ± 0.001 | 0.031 ± 0.000 | 0.083 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.198 ± 0.002 | 0.155 ± 0.001 | 0.827 ± 0.019 | 0.057 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.116 ± 0.001 | 0.062 ± 0.001 | 0.232 ± 0.003 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.041 ± 0.049 | 1.025 ± 0.041 | 0.904 ± 0.081 | 0.876 ± 0.026 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.117 ± 0.001 | 0.038 ± 0.001 | 0.035 ± 0.002 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.325 ± 0.014 | 0.276 ± 0.009 | 0.235 ± 0.010 | 0.208 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.152 ± 0.003 | 0.086 ± 0.002 | 0.073 ± 0.003 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.195 ± 0.005 | 0.153 ± 0.005 | 0.103 ± 0.003 | 0.101 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.104 ± 0.006 | 0.023 ± 0.001 | 0.022 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.128 ± 0.002 | 0.054 ± 0.001 | 0.042 ± 0.001 | 0.031 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.107 ± 0.001 | 0.029 ± 0.000 | 0.026 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.253 ± 0.003 | 0.210 ± 0.003 | 0.156 ± 0.005 | 0.149 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.101 ± 0.002 | 0.024 ± 0.001 | 0.023 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.143 ± 0.002 | 0.069 ± 0.000 | 0.055 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.111 ± 0.002 | 0.033 ± 0.000 | 0.029 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.413 ± 0.014 | 0.389 ± 0.008 | 0.313 ± 0.016 | 0.296 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.106 ± 0.001 | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.175 ± 0.003 | 0.114 ± 0.003 | 0.095 ± 0.004 | 0.077 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.123 ± 0.002 | 0.045 ± 0.001 | 0.039 ± 0.001 | 0.026 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 63.308 ± 0.267 | 63.118 ± 0.092 | 64.132 ± 0.451 | 67.439 ± 0.412 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.629 ± 0.004 | 0.628 ± 0.007 | 0.625 ± 0.006 | 0.653 ± 0.006 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.679 ± 0.014 | 2.667 ± 0.025 | 2.703 ± 0.037 | 2.796 ± 0.034 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 12.062 ± 0.093 | 11.992 ± 0.077 | 12.257 ± 0.067 | 12.761 ± 0.066 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.201 ± 0.001 | 0.204 ± 0.001 | 0.201 ± 0.002 | 0.206 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 71.144 ± 0.291 | 80.812 ± 0.601 | 134.101 ± 0.255 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 2.922 ± 0.031 | 3.051 ± 0.019 | 3.897 ± 0.015 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 13.298 ± 0.055 | 14.471 ± 0.098 | 21.305 ± 0.062 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 70.883 ± 0.762 | 69.317 ± 0.848 | 132.853 ± 0.189 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 2.876 ± 0.008 | 2.798 ± 0.023 | 3.884 ± 0.018 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 13.193 ± 0.052 | 12.916 ± 0.047 | 21.123 ± 0.126 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 32.001 ± 1.278 | 59.123 ± 0.989 | 85.911 ± 0.266 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.656 ± 0.003 | 1.295 ± 0.020 | 1.713 ± 0.005 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 3.983 ± 0.028 | 8.359 ± 0.037 | 11.859 ± 0.067 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 28.797 ± 0.903 | 30.872 ± 0.674 | 86.545 ± 0.784 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.678 ± 0.005 | 0.763 ± 0.007 | 1.722 ± 0.016 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 4.002 ± 0.075 | 4.585 ± 0.067 | 11.712 ± 0.045 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 5.201 ± 1.010 | 5.151 ± 1.242 | 5.600 ± 0.770 | 33.293 ± 0.151 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.031 ± 0.001 | 0.027 ± 0.000 | 0.018 ± 0.001 | 0.077 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 0.117 ± 0.004 | 0.104 ± 0.003 | 0.103 ± 0.003 | 0.562 ± 0.006 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 0.706 ± 0.004 | 0.657 ± 0.026 | 0.631 ± 0.012 | 4.232 ± 0.021 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | 16.577 ± 0.775 | 16.455 ± 0.452 | 16.323 ± 0.496 | 100.528 ± 0.537 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.126 ± 0.002 | 0.081 ± 0.001 | 0.059 ± 0.001 | 0.223 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | 0.397 ± 0.004 | 0.333 ± 0.003 | 0.294 ± 0.010 | 1.666 ± 0.021 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | 2.250 ± 0.057 | 2.127 ± 0.020 | 1.938 ± 0.063 | 12.647 ± 0.053 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.101 ± 0.001 | 0.033 ± 0.000 | 0.024 ± 0.001 | 0.036 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 49.817 ± 0.552 | 56.094 ± 0.538 | 134.247 ± 0.251 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.787 ± 0.010 | 1.909 ± 0.039 | 3.145 ± 0.021 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 8.538 ± 0.037 | 9.843 ± 0.055 | 18.737 ± 0.059 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 50.146 ± 0.598 | 53.597 ± 1.955 | 136.868 ± 3.204 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.829 ± 0.009 | 1.834 ± 0.011 | 3.122 ± 0.012 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 8.708 ± 0.019 | 9.324 ± 0.068 | 19.406 ± 0.083 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | 5.412 ± 0.121 | 5.070 ± 0.103 | 5.812 ± 0.117 | 5.103 ± 0.065 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.143 ± 0.002 | 0.078 ± 0.001 | 0.077 ± 0.003 | 0.061 ± 0.002 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | 0.304 ± 0.010 | 0.242 ± 0.003 | 0.256 ± 0.011 | 0.237 ± 0.011 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | 1.177 ± 0.042 | 1.063 ± 0.017 | 1.235 ± 0.032 | 1.025 ± 0.028 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.123 ± 0.002 | 0.040 ± 0.000 | 0.039 ± 0.001 | 0.021 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 4.753 ± 0.050 | 5.611 ± 0.085 | 5.165 ± 0.054 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.216 ± 0.011 | 0.293 ± 0.008 | 0.220 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.000 ± 0.045 | 1.182 ± 0.046 | 1.019 ± 0.016 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 4.629 ± 0.074 | 9.263 ± 0.041 | 4.752 ± 0.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.209 ± 0.001 | 0.387 ± 0.006 | 0.215 ± 0.002 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 0.942 ± 0.016 | 1.862 ± 0.010 | 0.978 ± 0.013 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | 116.525 ± 1.083 | 115.488 ± 0.629 | 112.287 ± 1.812 | 175.515 ± 1.598 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 1.304 ± 0.014 | 1.244 ± 0.008 | 1.254 ± 0.014 | 1.368 ± 0.012 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | 5.095 ± 0.045 | 5.010 ± 0.026 | 4.946 ± 0.038 | 6.009 ± 0.019 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | 23.029 ± 0.084 | 22.906 ± 0.052 | 22.530 ± 0.104 | 30.639 ± 0.134 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.351 ± 0.007 | 0.300 ± 0.004 | 0.298 ± 0.003 | 0.300 ± 0.001 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 116.999 ± 0.950 | 133.649 ± 1.495 | 176.614 ± 0.626 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.012 ± 0.036 | 5.563 ± 0.094 | 6.051 ± 0.024 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 22.796 ± 0.103 | 25.756 ± 0.142 | 30.771 ± 0.044 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 115.746 ± 0.731 | 112.015 ± 1.212 | 175.465 ± 0.389 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 5.009 ± 0.029 | 5.054 ± 0.063 | 6.009 ± 0.024 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 22.843 ± 0.096 | 22.482 ± 0.149 | 30.517 ± 0.051 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 4.523 ± 0.068 | 5.152 ± 0.696 | 4.931 ± 0.296 | 32.884 ± 0.151 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.001 | 0.069 ± 0.001 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.085 ± 0.001 | 0.101 ± 0.002 | 0.097 ± 0.002 | 0.537 ± 0.004 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.565 ± 0.009 | 0.658 ± 0.012 | 0.602 ± 0.018 | 4.120 ± 0.039 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.210 ± 0.012 | 1.284 ± 0.024 | 1.252 ± 0.031 | 8.075 ± 0.072 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.288 ± 0.005 | 0.369 ± 0.008 | 0.345 ± 0.017 | 2.145 ± 0.027 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 25.416 ± 0.105 | 25.246 ± 0.063 | 31.303 ± 0.428 | 27.152 ± 0.124 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.380 ± 0.002 | 0.379 ± 0.003 | 0.401 ± 0.004 | 0.384 ± 0.004 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.182 ± 0.067 | 1.181 ± 0.004 | 1.335 ± 0.016 | 1.237 ± 0.005 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 5.013 ± 0.026 | 5.009 ± 0.022 | 6.142 ± 0.068 | 5.381 ± 0.029 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.071 ± 0.000 | 0.072 ± 0.000 | 0.070 ± 0.001 | 0.072 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 4.083 ± 0.078 | 4.098 ± 0.030 | 5.195 ± 0.085 | 5.009 ± 0.029 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 4.018 ± 0.017 | 3.990 ± 0.061 | 5.117 ± 0.099 | 5.021 ± 0.023 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 4.474 ± 0.133 | 4.377 ± 0.065 | 5.519 ± 0.112 | 5.440 ± 0.060 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.039 ± 0.001 | 0.049 ± 0.001 | 0.049 ± 0.001 | 0.054 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.042 ± 0.000 | 0.045 ± 0.002 | 0.050 ± 0.004 | 0.082 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.063 ± 0.000 | 0.057 ± 0.000 | 0.062 ± 0.003 | 0.095 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.155 ± 0.002 | 0.164 ± 0.001 | 0.196 ± 0.005 | 0.208 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.157 ± 0.011 | 0.156 ± 0.003 | 0.191 ± 0.010 | 0.221 ± 0.012 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.202 ± 0.007 | 0.185 ± 0.008 | 0.224 ± 0.004 | 0.267 ± 0.012 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 0.806 ± 0.016 | 0.814 ± 0.015 | 1.039 ± 0.022 | 1.031 ± 0.051 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 0.798 ± 0.020 | 0.792 ± 0.022 | 1.042 ± 0.022 | 1.035 ± 0.025 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 0.931 ± 0.014 | 0.883 ± 0.012 | 1.160 ± 0.028 | 1.154 ± 0.039 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.017 ± 0.001 | 0.017 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.020 ± 0.001 | 0.019 ± 0.001 | 0.023 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.025 ± 0.002 | 0.049 ± 0.008 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 106.646 ± 0.321 | 106.267 ± 0.322 | 107.257 ± 0.346 | 107.474 ± 0.575 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.217 ± 0.005 | 1.215 ± 0.009 | 1.208 ± 0.005 | 1.231 ± 0.013 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.848 ± 0.017 | 4.832 ± 0.023 | 4.833 ± 0.045 | 4.862 ± 0.043 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 21.638 ± 0.050 | 21.569 ± 0.124 | 21.708 ± 0.101 | 21.992 ± 0.078 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.278 ± 0.002 | 0.278 ± 0.001 | 0.276 ± 0.001 | 0.277 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.042 ± 0.000 | 0.040 ± 0.003 | 0.040 ± 0.001 | 0.040 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.042 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.000 | 0.050 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.001 | 0.033 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.064 ± 0.001 | 0.074 ± 0.001 | 0.045 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.032 ± 0.000 | 0.043 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.045 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.063 ± 0.000 | 0.068 ± 0.001 | 0.045 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.001 | 0.037 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.033 ± 0.000 | 0.061 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.038 ± 0.000 | 0.075 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.024 ± 0.000 | 0.057 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.032 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.038 ± 0.000 | 0.056 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.001 | 0.039 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | 0.090 ± 0.001 | 0.021 ± 0.000 | 0.016 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.101 ± 0.001 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | 0.091 ± 0.001 | 0.022 ± 0.000 | 0.017 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.102 ± 0.001 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.082 ± 0.001 | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.050 ± 0.000 | 0.048 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.064 ± 0.000 | 0.054 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.040 ± 0.001 | 0.046 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.050 ± 0.000 | 0.047 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.040 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.065 ± 0.001 | 0.054 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.040 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.047 ± 0.000 | 0.045 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | 0.099 ± 0.001 | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.096 ± 0.003 | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | 0.113 ± 0.001 | 0.030 ± 0.000 | 0.027 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.097 ± 0.002 | 0.020 ± 0.001 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.097 ± 0.003 | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.017 ± 0.000 | 0.082 ± 0.004 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.081 ± 0.005 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.016 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.016 ± 0.000 | 0.038 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.023 ± 0.000 | 0.049 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | 0.109 ± 0.001 | 0.039 ± 0.001 | 0.040 ± 0.001 | 0.028 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.082 ± 0.002 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | 0.156 ± 0.000 | 0.079 ± 0.001 | 0.082 ± 0.003 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.085 ± 0.002 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.088 ± 0.002 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.000 | 0.075 ± 0.001 | 0.027 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.079 ± 0.001 | 0.122 ± 0.001 | 0.074 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.015 ± 0.000 | 0.048 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.022 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.026 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.078 ± 0.002 | 0.102 ± 0.001 | 0.075 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.001 | 0.035 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.022 ± 0.000 | 0.025 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.072 ± 0.000 | 0.073 ± 0.000 | 0.069 ± 0.001 | 0.071 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 3.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`16x16`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`32x32`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_204226/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_204226`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_204226`.

- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_204226/cpu_ops_t4_20260923_204226.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_204226/cpu_ops_t4_20260923_204226.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 9.845 ± 0.070 | 9.976 ± 0.072 | 9.974 ± 0.203 | 5.275 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.158 ± 0.003 | 0.161 ± 0.003 | 0.160 ± 0.006 | 0.145 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.491 ± 0.024 | 2.495 ± 0.011 | 2.513 ± 0.134 | 1.172 ± 0.178 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.622 ± 0.007 | 0.628 ± 0.005 | 0.632 ± 0.025 | 0.259 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.125 ± 0.004 | 0.123 ± 0.001 | 0.133 ± 0.011 | 0.264 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.034 ± 0.001 | 0.035 ± 0.001 | 0.036 ± 0.001 | 0.071 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.020 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.694 ± 0.021 | 0.675 ± 0.007 | 0.701 ± 0.013 | 0.535 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.172 ± 0.002 | 0.172 ± 0.002 | 0.187 ± 0.023 | 0.216 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.001 | 0.046 ± 0.001 | 0.046 ± 0.002 | 0.057 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 2.820 ± 0.141 | 2.595 ± 0.029 | 2.667 ± 0.200 | 1.253 ± 0.172 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.043 ± 0.001 | 0.044 ± 0.000 | 0.046 ± 0.004 | 0.048 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.642 ± 0.012 | 0.651 ± 0.009 | 0.658 ± 0.010 | 0.330 ± 0.045 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.163 ± 0.002 | 0.165 ± 0.001 | 0.167 ± 0.009 | 0.180 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.248 ± 0.010 | 0.305 ± 0.013 | 1.067 ± 0.123 | 0.214 ± 0.014 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.024 ± 0.003 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.067 ± 0.002 | 0.091 ± 0.003 | 0.253 ± 0.015 | 0.090 ± 0.025 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.027 ± 0.001 | 0.071 ± 0.015 | 0.048 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.001 | 0.033 ± 0.000 | 0.008 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.038 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.038 ± 0.002 | 0.047 ± 0.002 | 0.029 ± 0.002 | 0.091 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.008 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.093 ± 0.011 | 0.113 ± 0.002 | 0.493 ± 0.037 | 0.063 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.023 ± 0.000 | 0.030 ± 0.001 | 0.127 ± 0.007 | 0.048 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.827 ± 0.033 | 1.802 ± 0.016 | 1.733 ± 0.089 | 1.030 ± 0.023 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.033 ± 0.000 | 0.034 ± 0.000 | 0.046 ± 0.002 | 0.035 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.468 ± 0.011 | 0.468 ± 0.010 | 0.477 ± 0.019 | 0.282 ± 0.028 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.120 ± 0.001 | 0.120 ± 0.001 | 0.130 ± 0.006 | 0.195 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.124 ± 0.001 | 0.124 ± 0.002 | 0.127 ± 0.016 | 0.161 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.034 ± 0.000 | 0.042 ± 0.001 | 0.045 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.002 | 0.014 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.274 ± 0.004 | 0.275 ± 0.004 | 0.278 ± 0.008 | 0.329 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.070 ± 0.001 | 0.071 ± 0.001 | 0.083 ± 0.003 | 0.087 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.034 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.675 ± 0.009 | 0.676 ± 0.009 | 0.685 ± 0.018 | 0.378 ± 0.025 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.171 ± 0.002 | 0.177 ± 0.002 | 0.186 ± 0.009 | 0.192 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.046 ± 0.001 | 0.047 ± 0.000 | 0.057 ± 0.002 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.553 ± 0.017 | 0.566 ± 0.019 | 0.295 ± 0.039 | 0.512 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.027 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.151 ± 0.002 | 0.153 ± 0.001 | 0.108 ± 0.007 | 0.151 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.041 ± 0.001 | 0.042 ± 0.001 | 0.042 ± 0.002 | 0.077 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.056 ± 0.001 | 0.056 ± 0.001 | 0.059 ± 0.005 | 0.077 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.018 ± 0.001 | 0.018 ± 0.000 | 0.032 ± 0.003 | 0.024 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.096 ± 0.010 | 0.089 ± 0.001 | 0.076 ± 0.006 | 0.120 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.025 ± 0.001 | 0.027 ± 0.001 | 0.036 ± 0.001 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.023 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.239 ± 0.026 | 0.207 ± 0.004 | 0.133 ± 0.004 | 0.172 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.053 ± 0.002 | 0.055 ± 0.002 | 0.050 ± 0.002 | 0.069 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | 0.020 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 20.061 ± 0.144 | 20.186 ± 0.106 | 20.139 ± 0.114 | 10.535 ± 0.038 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.322 ± 0.009 | 0.322 ± 0.004 | 0.320 ± 0.003 | 0.191 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.038 ± 0.054 | 5.055 ± 0.056 | 5.146 ± 0.089 | 2.383 ± 0.322 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.279 ± 0.014 | 1.266 ± 0.022 | 1.280 ± 0.035 | 0.454 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.314 ± 0.007 | 0.312 ± 0.002 | 0.317 ± 0.008 | 1.088 ± 0.053 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.021 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.081 ± 0.002 | 0.083 ± 0.001 | 0.084 ± 0.002 | 0.264 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.024 ± 0.004 | 0.072 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.349 ± 0.026 | 1.367 ± 0.015 | 1.388 ± 0.031 | 0.684 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.026 ± 0.000 | 0.027 ± 0.000 | 0.025 ± 0.001 | 0.037 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.341 ± 0.005 | 0.345 ± 0.006 | 0.347 ± 0.012 | 0.522 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.088 ± 0.002 | 0.091 ± 0.001 | 0.090 ± 0.004 | 0.135 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 5.130 ± 0.119 | 5.157 ± 0.064 | 5.109 ± 0.073 | 2.226 ± 0.705 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.084 ± 0.001 | 0.087 ± 0.001 | 0.084 ± 0.005 | 0.097 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.279 ± 0.014 | 1.288 ± 0.010 | 1.298 ± 0.017 | 0.487 ± 0.014 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.323 ± 0.007 | 0.328 ± 0.003 | 0.328 ± 0.005 | 0.296 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.917 ± 0.017 | 0.857 ± 0.022 | 3.911 ± 0.430 | 0.581 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.109 ± 0.001 | 0.038 ± 0.000 | 0.097 ± 0.005 | 0.018 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.317 ± 0.007 | 0.300 ± 0.004 | 1.020 ± 0.042 | 0.189 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.137 ± 0.002 | 0.093 ± 0.001 | 0.276 ± 0.014 | 0.062 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.173 ± 0.001 | 0.106 ± 0.001 | 0.040 ± 0.002 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.082 ± 0.002 | 0.017 ± 0.000 | 0.030 ± 0.001 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.103 ± 0.001 | 0.038 ± 0.000 | 0.033 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.085 ± 0.002 | 0.021 ± 0.000 | 0.032 ± 0.003 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.225 ± 0.010 | 0.148 ± 0.003 | 0.103 ± 0.005 | 0.257 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.082 ± 0.002 | 0.018 ± 0.000 | 0.030 ± 0.004 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.116 ± 0.002 | 0.050 ± 0.001 | 0.042 ± 0.003 | 0.070 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.088 ± 0.001 | 0.024 ± 0.000 | 0.032 ± 0.002 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.594 ± 0.010 | 0.573 ± 0.016 | 3.303 ± 0.027 | 0.203 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.094 ± 0.001 | 0.030 ± 0.000 | 0.081 ± 0.009 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.200 ± 0.004 | 0.155 ± 0.001 | 0.828 ± 0.027 | 0.062 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.117 ± 0.002 | 0.061 ± 0.000 | 0.236 ± 0.008 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.991 ± 0.039 | 1.033 ± 0.010 | 0.510 ± 0.054 | 0.659 ± 0.027 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.118 ± 0.001 | 0.038 ± 0.001 | 0.052 ± 0.003 | 0.020 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.331 ± 0.005 | 0.304 ± 0.004 | 0.218 ± 0.016 | 0.186 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.152 ± 0.004 | 0.088 ± 0.001 | 0.081 ± 0.007 | 0.089 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.196 ± 0.004 | 0.154 ± 0.002 | 0.106 ± 0.004 | 0.106 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.104 ± 0.003 | 0.023 ± 0.000 | 0.039 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.128 ± 0.002 | 0.054 ± 0.001 | 0.058 ± 0.003 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.107 ± 0.002 | 0.030 ± 0.000 | 0.045 ± 0.003 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.280 ± 0.013 | 0.213 ± 0.002 | 0.135 ± 0.008 | 0.156 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.104 ± 0.002 | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.144 ± 0.002 | 0.070 ± 0.001 | 0.066 ± 0.003 | 0.045 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.112 ± 0.002 | 0.033 ± 0.000 | 0.048 ± 0.004 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.498 ± 0.059 | 0.412 ± 0.008 | 0.251 ± 0.009 | 0.229 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.107 ± 0.001 | 0.027 ± 0.000 | 0.046 ± 0.003 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.177 ± 0.003 | 0.115 ± 0.003 | 0.091 ± 0.004 | 0.084 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.125 ± 0.002 | 0.046 ± 0.000 | 0.055 ± 0.002 | 0.028 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 62.439 ± 0.232 | 62.811 ± 0.261 | 65.155 ± 0.722 | 66.507 ± 1.459 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.609 ± 0.005 | 0.625 ± 0.008 | 0.622 ± 0.009 | 0.657 ± 0.006 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.614 ± 0.023 | 2.659 ± 0.028 | 2.740 ± 0.080 | 2.780 ± 0.059 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 11.872 ± 0.042 | 11.925 ± 0.050 | 12.293 ± 0.234 | 13.077 ± 0.944 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.199 ± 0.002 | 0.202 ± 0.001 | 0.205 ± 0.003 | 0.210 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 68.940 ± 1.360 | 87.304 ± 4.373 | 86.684 ± 2.318 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 2.922 ± 0.031 | 3.095 ± 0.058 | 3.094 ± 0.052 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 12.914 ± 0.074 | 14.453 ± 0.179 | 15.311 ± 0.157 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 68.503 ± 0.468 | 72.967 ± 1.888 | 87.198 ± 4.028 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 2.898 ± 0.018 | 2.833 ± 0.083 | 3.040 ± 0.019 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 12.811 ± 0.050 | 12.822 ± 0.348 | 15.126 ± 0.211 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 26.616 ± 0.251 | 70.549 ± 11.496 | 36.209 ± 1.115 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.702 ± 0.003 | 1.402 ± 0.100 | 0.770 ± 0.024 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 3.926 ± 0.022 | 8.110 ± 0.683 | 5.382 ± 0.152 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 27.160 ± 0.236 | 31.979 ± 4.268 | 36.525 ± 0.775 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.708 ± 0.007 | 0.863 ± 0.039 | 0.828 ± 0.026 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.038 ± 0.034 | 4.401 ± 0.310 | 5.475 ± 0.123 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 5.760 ± 0.315 | 4.881 ± 0.669 | 6.035 ± 1.245 | 9.813 ± 1.672 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.030 ± 0.002 | 0.027 ± 0.000 | 0.019 ± 0.001 | 0.080 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.129 ± 0.002 | 0.123 ± 0.003 | 0.121 ± 0.013 | 0.190 ± 0.006 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 0.714 ± 0.008 | 0.643 ± 0.025 | 0.665 ± 0.031 | 1.224 ± 0.028 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.001 | 0.014 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | 16.163 ± 1.006 | 15.648 ± 0.381 | 16.620 ± 1.395 | 27.727 ± 0.620 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.126 ± 0.002 | 0.081 ± 0.001 | 0.059 ± 0.003 | 0.189 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | 0.418 ± 0.004 | 0.370 ± 0.006 | 0.340 ± 0.046 | 0.591 ± 0.125 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | 2.170 ± 0.040 | 2.019 ± 0.031 | 2.171 ± 0.129 | 3.576 ± 0.324 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.102 ± 0.001 | 0.033 ± 0.000 | 0.024 ± 0.001 | 0.037 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 51.812 ± 0.964 | 53.852 ± 2.599 | 73.126 ± 13.614 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 1.819 ± 0.006 | 1.892 ± 0.101 | 1.854 ± 0.144 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 7.881 ± 0.024 | 8.647 ± 0.191 | 10.108 ± 0.192 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 51.117 ± 1.591 | 53.234 ± 4.133 | 66.130 ± 1.675 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 1.829 ± 0.016 | 1.917 ± 0.135 | 1.904 ± 0.062 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 7.997 ± 0.060 | 8.543 ± 0.268 | 10.383 ± 0.152 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | 5.244 ± 0.071 | 4.922 ± 0.106 | 6.040 ± 0.892 | 4.751 ± 0.656 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.143 ± 0.002 | 0.079 ± 0.001 | 0.083 ± 0.007 | 0.065 ± 0.002 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | 0.323 ± 0.009 | 0.272 ± 0.003 | 0.253 ± 0.009 | 0.228 ± 0.025 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | 1.122 ± 0.092 | 1.039 ± 0.009 | 1.216 ± 0.098 | 1.002 ± 0.037 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.123 ± 0.001 | 0.040 ± 0.001 | 0.039 ± 0.002 | 0.022 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 4.549 ± 0.061 | 5.129 ± 0.058 | 4.955 ± 0.194 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.236 ± 0.005 | 0.296 ± 0.010 | 0.230 ± 0.019 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 0.956 ± 0.008 | 1.079 ± 0.082 | 1.143 ± 0.029 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 4.447 ± 0.037 | 8.695 ± 0.326 | 4.692 ± 0.099 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.224 ± 0.001 | 0.407 ± 0.010 | 0.216 ± 0.023 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 0.933 ± 0.011 | 1.782 ± 0.082 | 1.086 ± 0.011 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | 115.367 ± 0.207 | 114.838 ± 0.505 | 119.401 ± 5.964 | 130.762 ± 4.375 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 1.296 ± 0.013 | 1.244 ± 0.006 | 1.245 ± 0.010 | 1.375 ± 0.019 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | 5.062 ± 0.022 | 5.044 ± 0.038 | 5.050 ± 0.085 | 5.406 ± 0.201 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | 22.986 ± 0.059 | 22.777 ± 0.180 | 23.351 ± 0.258 | 24.829 ± 0.556 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.350 ± 0.007 | 0.297 ± 0.002 | 0.299 ± 0.004 | 0.310 ± 0.006 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 131.924 ± 32.161 | 137.546 ± 18.473 | 130.580 ± 6.297 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 4.994 ± 0.027 | 6.106 ± 0.239 | 5.333 ± 0.082 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 22.825 ± 0.028 | 25.802 ± 0.511 | 25.050 ± 0.820 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 120.989 ± 5.414 | 117.604 ± 3.132 | 132.051 ± 6.428 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 4.973 ± 0.020 | 5.396 ± 0.207 | 5.386 ± 0.263 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 22.762 ± 0.065 | 23.147 ± 0.276 | 24.504 ± 0.706 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.504 ± 0.068 | 4.695 ± 0.253 | 5.766 ± 2.339 | 8.551 ± 0.233 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.015 ± 0.001 | 0.073 ± 0.002 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.084 ± 0.000 | 0.112 ± 0.002 | 0.100 ± 0.002 | 0.173 ± 0.019 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.564 ± 0.002 | 0.643 ± 0.011 | 0.629 ± 0.041 | 1.162 ± 0.042 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.205 ± 0.017 | 1.265 ± 0.015 | 1.251 ± 0.032 | 2.112 ± 0.025 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.292 ± 0.003 | 0.361 ± 0.011 | 0.339 ± 0.026 | 0.616 ± 0.042 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 25.301 ± 0.059 | 25.205 ± 0.048 | 29.036 ± 1.424 | 25.867 ± 0.232 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.377 ± 0.003 | 0.376 ± 0.001 | 0.410 ± 0.014 | 0.388 ± 0.006 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.185 ± 0.007 | 1.182 ± 0.002 | 1.245 ± 0.023 | 1.246 ± 0.011 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 4.997 ± 0.015 | 4.994 ± 0.033 | 5.388 ± 0.246 | 5.439 ± 0.469 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.071 ± 0.000 | 0.071 ± 0.000 | 0.086 ± 0.005 | 0.073 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 4.087 ± 0.037 | 4.060 ± 0.062 | 4.840 ± 0.242 | 4.386 ± 0.034 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 4.033 ± 0.044 | 3.961 ± 0.038 | 5.096 ± 0.236 | 4.410 ± 0.041 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 4.436 ± 0.048 | 4.329 ± 0.042 | 5.519 ± 0.546 | 4.762 ± 0.033 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.038 ± 0.001 | 0.043 ± 0.001 | 0.052 ± 0.003 | 0.056 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.042 ± 0.000 | 0.044 ± 0.002 | 0.054 ± 0.008 | 0.082 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.063 ± 0.000 | 0.060 ± 0.000 | 0.075 ± 0.014 | 0.094 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.151 ± 0.002 | 0.169 ± 0.001 | 0.187 ± 0.012 | 0.206 ± 0.014 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.153 ± 0.002 | 0.166 ± 0.004 | 0.205 ± 0.020 | 0.261 ± 0.037 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.202 ± 0.004 | 0.194 ± 0.004 | 0.257 ± 0.021 | 0.346 ± 0.120 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 0.802 ± 0.018 | 0.794 ± 0.005 | 0.968 ± 0.054 | 0.959 ± 0.055 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 0.787 ± 0.035 | 0.783 ± 0.006 | 0.928 ± 0.037 | 0.972 ± 0.015 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 0.929 ± 0.041 | 0.879 ± 0.012 | 1.030 ± 0.025 | 1.084 ± 0.023 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.001 | 0.019 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.020 ± 0.000 | 0.019 ± 0.001 | 0.026 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.026 ± 0.000 | 0.025 ± 0.001 | 0.024 ± 0.001 | 0.052 ± 0.004 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 106.125 ± 0.772 | 105.674 ± 0.360 | 110.251 ± 7.643 | 108.633 ± 3.928 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.216 ± 0.008 | 1.210 ± 0.009 | 1.265 ± 0.041 | 1.242 ± 0.016 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.845 ± 0.021 | 4.809 ± 0.015 | 4.967 ± 0.190 | 4.844 ± 0.041 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 21.664 ± 0.056 | 21.533 ± 0.020 | 23.378 ± 0.344 | 22.378 ± 1.393 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.277 ± 0.002 | 0.276 ± 0.002 | 0.279 ± 0.003 | 0.287 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.042 ± 0.000 | 0.042 ± 0.000 | 0.039 ± 0.002 | 0.041 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.041 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.040 ± 0.000 | 0.051 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.065 ± 0.001 | 0.078 ± 0.003 | 0.044 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.000 | 0.034 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.032 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.000 | 0.043 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.064 ± 0.000 | 0.071 ± 0.002 | 0.044 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.033 ± 0.000 | 0.142 ± 0.010 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.000 | 0.131 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.039 ± 0.001 | 0.146 ± 0.008 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.130 ± 0.005 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.137 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.033 ± 0.000 | 0.106 ± 0.008 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.000 | 0.093 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.038 ± 0.000 | 0.114 ± 0.009 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.093 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.032 ± 0.000 | 0.099 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | 0.090 ± 0.000 | 0.021 ± 0.000 | 0.016 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.077 ± 0.001 | 0.015 ± 0.001 | 0.011 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | 0.092 ± 0.001 | 0.022 ± 0.000 | 0.018 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.077 ± 0.001 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.083 ± 0.001 | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.051 ± 0.001 | 0.077 ± 0.007 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.040 ± 0.000 | 0.072 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.065 ± 0.001 | 0.084 ± 0.003 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.040 ± 0.000 | 0.072 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.001 | 0.074 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.051 ± 0.002 | 0.085 ± 0.013 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.041 ± 0.001 | 0.072 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.066 ± 0.000 | 0.091 ± 0.011 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.040 ± 0.001 | 0.071 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.048 ± 0.002 | 0.073 ± 0.003 | 0.005 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | 0.100 ± 0.001 | 0.021 ± 0.001 | 0.021 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.098 ± 0.000 | 0.020 ± 0.001 | 0.019 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | 0.114 ± 0.003 | 0.031 ± 0.000 | 0.028 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.098 ± 0.001 | 0.020 ± 0.000 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.100 ± 0.001 | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.017 ± 0.000 | 0.086 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.024 ± 0.000 | 0.095 ± 0.008 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.016 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.016 ± 0.000 | 0.082 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.016 ± 0.000 | 0.040 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.015 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.023 ± 0.000 | 0.051 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.016 ± 0.000 | 0.037 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | 0.112 ± 0.000 | 0.039 ± 0.000 | 0.039 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.084 ± 0.002 | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | 0.158 ± 0.002 | 0.079 ± 0.001 | 0.084 ± 0.002 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.086 ± 0.001 | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.091 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.040 ± 0.000 | 0.075 ± 0.002 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.048 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.080 ± 0.001 | 0.125 ± 0.004 | 0.075 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.028 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.013 ± 0.000 | 0.031 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.081 ± 0.003 | 0.103 ± 0.004 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.014 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.018 ± 0.000 | 0.036 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.009 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.024 ± 0.000 | 0.025 ± 0.001 | 0.024 ± 0.001 | 0.025 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.072 ± 0.000 | 0.072 ± 0.000 | 0.070 ± 0.002 | 0.071 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 3.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 28.4x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 35.5x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 32.6x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`16x16`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`32x32`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.1x faster than the slowest successful cell (`tenferro-eager`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.2x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 23.3x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

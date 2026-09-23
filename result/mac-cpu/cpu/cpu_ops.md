# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`
- Benchmark commit: `9f63fb7704feec767fda97251865741714f3926f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_173045/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_173045`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_173045`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_173045/cpu_ops_t1_20260923_173045.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_173045/cpu_ops_t1_20260923_173045.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 9.884 ± 0.039 | 9.925 ± 0.065 | 10.023 ± 0.042 | 10.739 ± 0.353 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.158 ± 0.002 | 0.161 ± 0.003 | 0.158 ± 0.001 | 0.164 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.462 ± 0.008 | 2.479 ± 0.032 | 2.574 ± 0.076 | 2.642 ± 0.065 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.624 ± 0.007 | 0.624 ± 0.026 | 0.654 ± 0.036 | 0.650 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.122 ± 0.005 | 0.124 ± 0.005 | 0.136 ± 0.004 | 0.266 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.034 ± 0.001 | 0.034 ± 0.002 | 0.036 ± 0.002 | 0.070 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.671 ± 0.009 | 0.672 ± 0.015 | 0.700 ± 0.008 | 0.842 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.172 ± 0.004 | 0.171 ± 0.003 | 0.172 ± 0.003 | 0.212 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.046 ± 0.001 | 0.047 ± 0.001 | 0.045 ± 0.001 | 0.056 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 2.573 ± 0.051 | 2.582 ± 0.081 | 2.604 ± 0.013 | 2.802 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.044 ± 0.001 | 0.044 ± 0.001 | 0.042 ± 0.000 | 0.047 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.649 ± 0.008 | 0.648 ± 0.008 | 0.648 ± 0.008 | 0.710 ± 0.021 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.162 ± 0.002 | 0.164 ± 0.003 | 0.164 ± 0.002 | 0.182 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.261 ± 0.012 | 0.309 ± 0.001 | 1.058 ± 0.043 | 0.383 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.023 ± 0.001 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.068 ± 0.002 | 0.081 ± 0.001 | 0.269 ± 0.009 | 0.082 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.026 ± 0.001 | 0.070 ± 0.004 | 0.048 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.001 | 0.033 ± 0.001 | 0.008 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.037 ± 0.002 | 0.045 ± 0.001 | 0.021 ± 0.005 | 0.050 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.008 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.087 ± 0.001 | 0.103 ± 0.006 | 0.496 ± 0.017 | 0.081 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.023 ± 0.001 | 0.030 ± 0.001 | 0.126 ± 0.005 | 0.049 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.036 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.830 ± 0.025 | 1.856 ± 0.053 | 1.847 ± 0.038 | 2.035 ± 0.098 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.033 ± 0.000 | 0.033 ± 0.001 | 0.031 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.462 ± 0.009 | 0.464 ± 0.008 | 0.477 ± 0.016 | 0.523 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.119 ± 0.002 | 0.119 ± 0.002 | 0.120 ± 0.002 | 0.127 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.124 ± 0.001 | 0.125 ± 0.002 | 0.122 ± 0.003 | 0.164 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.034 ± 0.001 | 0.032 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.276 ± 0.002 | 0.274 ± 0.004 | 0.280 ± 0.006 | 0.326 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.071 ± 0.001 | 0.071 ± 0.001 | 0.071 ± 0.001 | 0.085 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.674 ± 0.015 | 0.687 ± 0.018 | 0.684 ± 0.008 | 0.755 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.174 ± 0.003 | 0.176 ± 0.004 | 0.171 ± 0.004 | 0.195 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.046 ± 0.001 | 0.048 ± 0.002 | 0.044 ± 0.001 | 0.052 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.555 ± 0.011 | 0.576 ± 0.018 | 0.607 ± 0.021 | 0.793 ± 0.060 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.014 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.145 ± 0.006 | 0.147 ± 0.009 | 0.160 ± 0.002 | 0.190 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.040 ± 0.001 | 0.042 ± 0.001 | 0.044 ± 0.002 | 0.049 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.055 ± 0.001 | 0.055 ± 0.001 | 0.070 ± 0.013 | 0.076 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.089 ± 0.003 | 0.094 ± 0.002 | 0.098 ± 0.003 | 0.118 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.026 ± 0.001 | 0.027 ± 0.001 | 0.028 ± 0.001 | 0.034 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.201 ± 0.005 | 0.207 ± 0.009 | 0.216 ± 0.007 | 0.254 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.054 ± 0.001 | 0.058 ± 0.003 | 0.057 ± 0.002 | 0.068 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 19.848 ± 0.113 | 19.952 ± 0.072 | 20.287 ± 0.080 | 21.090 ± 0.298 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.320 ± 0.003 | 0.319 ± 0.004 | 0.320 ± 0.003 | 0.331 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.015 ± 0.147 | 4.999 ± 0.058 | 5.173 ± 0.044 | 5.257 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.251 ± 0.013 | 1.256 ± 0.007 | 1.308 ± 0.017 | 1.314 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.312 ± 0.004 | 0.312 ± 0.009 | 0.330 ± 0.009 | 1.082 ± 0.080 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.081 ± 0.001 | 0.081 ± 0.002 | 0.081 ± 0.002 | 0.256 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.023 ± 0.001 | 0.073 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.354 ± 0.023 | 1.359 ± 0.029 | 1.385 ± 0.018 | 2.099 ± 0.040 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.026 ± 0.001 | 0.027 ± 0.000 | 0.024 ± 0.000 | 0.038 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.349 ± 0.003 | 0.343 ± 0.010 | 0.345 ± 0.005 | 0.509 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.088 ± 0.002 | 0.091 ± 0.002 | 0.088 ± 0.001 | 0.135 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 5.120 ± 0.056 | 5.049 ± 0.128 | 5.213 ± 0.033 | 5.850 ± 0.073 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.085 ± 0.002 | 0.086 ± 0.001 | 0.084 ± 0.001 | 0.094 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.283 ± 0.021 | 1.280 ± 0.012 | 1.293 ± 0.023 | 1.489 ± 0.075 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.324 ± 0.006 | 0.330 ± 0.008 | 0.326 ± 0.002 | 0.363 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.948 ± 0.029 | 0.950 ± 0.007 | 3.881 ± 0.121 | 0.936 ± 0.018 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.108 ± 0.001 | 0.035 ± 0.000 | 0.094 ± 0.005 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.297 ± 0.003 | 0.255 ± 0.003 | 0.979 ± 0.062 | 0.232 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.136 ± 0.002 | 0.092 ± 0.001 | 0.278 ± 0.009 | 0.057 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.172 ± 0.002 | 0.105 ± 0.001 | 0.052 ± 0.007 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.080 ± 0.002 | 0.017 ± 0.001 | 0.028 ± 0.000 | 0.005 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.103 ± 0.002 | 0.038 ± 0.000 | 0.031 ± 0.000 | 0.006 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.085 ± 0.002 | 0.021 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.199 ± 0.003 | 0.150 ± 0.002 | 0.075 ± 0.004 | 0.179 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.081 ± 0.002 | 0.018 ± 0.000 | 0.030 ± 0.001 | 0.006 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.115 ± 0.002 | 0.050 ± 0.001 | 0.040 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.088 ± 0.001 | 0.024 ± 0.000 | 0.032 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.536 ± 0.006 | 0.534 ± 0.009 | 3.242 ± 0.116 | 0.235 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.094 ± 0.001 | 0.031 ± 0.000 | 0.082 ± 0.003 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.197 ± 0.003 | 0.155 ± 0.002 | 0.828 ± 0.012 | 0.060 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.116 ± 0.002 | 0.061 ± 0.001 | 0.233 ± 0.004 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.033 ± 0.034 | 1.034 ± 0.035 | 0.924 ± 0.037 | 0.965 ± 0.019 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.118 ± 0.001 | 0.039 ± 0.001 | 0.035 ± 0.001 | 0.021 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.320 ± 0.013 | 0.277 ± 0.010 | 0.239 ± 0.011 | 0.227 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.171 ± 0.023 | 0.087 ± 0.003 | 0.079 ± 0.002 | 0.059 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.195 ± 0.002 | 0.154 ± 0.004 | 0.134 ± 0.013 | 0.105 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.101 ± 0.002 | 0.023 ± 0.000 | 0.023 ± 0.001 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.129 ± 0.001 | 0.054 ± 0.001 | 0.043 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.109 ± 0.005 | 0.029 ± 0.001 | 0.027 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.253 ± 0.003 | 0.216 ± 0.006 | 0.158 ± 0.011 | 0.154 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.103 ± 0.001 | 0.024 ± 0.001 | 0.023 ± 0.000 | 0.006 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.144 ± 0.002 | 0.069 ± 0.001 | 0.055 ± 0.001 | 0.044 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.112 ± 0.002 | 0.033 ± 0.001 | 0.030 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.411 ± 0.016 | 0.393 ± 0.013 | 0.320 ± 0.008 | 0.316 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.107 ± 0.003 | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.176 ± 0.004 | 0.114 ± 0.003 | 0.097 ± 0.004 | 0.081 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.123 ± 0.002 | 0.045 ± 0.001 | 0.041 ± 0.001 | 0.027 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 62.979 ± 0.164 | 63.265 ± 0.190 | 64.401 ± 0.575 | 69.516 ± 2.513 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.623 ± 0.006 | 0.621 ± 0.016 | 0.624 ± 0.005 | 0.660 ± 0.010 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.670 ± 0.028 | 2.675 ± 0.023 | 2.714 ± 0.047 | 2.915 ± 0.108 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 11.947 ± 0.064 | 11.929 ± 0.029 | 12.313 ± 0.137 | 12.984 ± 0.153 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.202 ± 0.001 | 0.203 ± 0.001 | 0.201 ± 0.002 | 0.209 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 74.312 ± 5.759 | 81.558 ± 1.560 | 135.062 ± 0.430 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 2.935 ± 0.037 | 3.048 ± 0.032 | 3.971 ± 0.204 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 13.324 ± 0.033 | 14.686 ± 0.289 | 22.272 ± 2.170 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 71.536 ± 2.843 | 70.180 ± 0.565 | 133.775 ± 0.362 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 2.912 ± 0.042 | 2.809 ± 0.018 | 3.933 ± 0.034 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 13.203 ± 0.069 | 13.336 ± 0.957 | 21.693 ± 0.093 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 28.572 ± 0.915 | 60.216 ± 2.437 | 86.434 ± 0.223 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.661 ± 0.007 | 1.303 ± 0.020 | 1.732 ± 0.023 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 3.969 ± 0.070 | 9.260 ± 1.334 | 12.048 ± 0.243 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 28.163 ± 0.419 | 37.567 ± 3.016 | 87.997 ± 0.545 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.683 ± 0.003 | 0.768 ± 0.006 | 1.740 ± 0.021 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 3.992 ± 0.038 | 4.701 ± 0.024 | 11.948 ± 0.207 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 5.338 ± 1.290 | 5.312 ± 0.996 | 6.195 ± 0.982 | 34.180 ± 0.873 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.031 ± 0.000 | 0.027 ± 0.000 | 0.018 ± 0.001 | 0.079 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 0.116 ± 0.002 | 0.104 ± 0.002 | 0.102 ± 0.003 | 0.570 ± 0.009 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 0.710 ± 0.010 | 0.654 ± 0.010 | 0.655 ± 0.026 | 4.389 ± 0.052 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | 16.926 ± 1.137 | 16.894 ± 1.090 | 16.766 ± 1.267 | 101.791 ± 0.832 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.127 ± 0.004 | 0.081 ± 0.001 | 0.058 ± 0.002 | 0.225 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | 0.395 ± 0.006 | 0.332 ± 0.006 | 0.297 ± 0.008 | 1.687 ± 0.013 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | 2.239 ± 0.045 | 2.127 ± 0.037 | 1.967 ± 0.101 | 13.127 ± 0.639 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.103 ± 0.001 | 0.033 ± 0.001 | 0.024 ± 0.000 | 0.036 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 48.876 ± 0.536 | 57.464 ± 2.067 | 135.567 ± 0.526 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.789 ± 0.024 | 1.939 ± 0.015 | 3.159 ± 0.035 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 8.462 ± 0.036 | 9.913 ± 0.065 | 18.956 ± 0.555 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 50.159 ± 3.571 | 54.754 ± 1.653 | 138.116 ± 0.214 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.835 ± 0.019 | 1.844 ± 0.012 | 3.145 ± 0.020 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 8.656 ± 0.045 | 9.937 ± 0.402 | 19.660 ± 0.075 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | 5.452 ± 0.135 | 5.133 ± 0.296 | 5.897 ± 0.159 | 5.886 ± 0.986 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.143 ± 0.001 | 0.078 ± 0.001 | 0.078 ± 0.003 | 0.063 ± 0.001 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | 0.315 ± 0.020 | 0.245 ± 0.004 | 0.273 ± 0.018 | 0.240 ± 0.008 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | 1.160 ± 0.036 | 1.056 ± 0.010 | 1.357 ± 0.078 | 1.061 ± 0.046 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.124 ± 0.001 | 0.040 ± 0.000 | 0.039 ± 0.001 | 0.022 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 4.738 ± 0.018 | 6.564 ± 0.188 | 5.205 ± 0.095 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.215 ± 0.003 | 0.299 ± 0.020 | 0.221 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 0.988 ± 0.015 | 1.194 ± 0.051 | 1.078 ± 0.058 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 4.579 ± 0.041 | 11.059 ± 0.628 | 4.777 ± 0.038 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.210 ± 0.003 | 0.385 ± 0.010 | 0.220 ± 0.008 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 0.944 ± 0.018 | 1.897 ± 0.058 | 0.991 ± 0.047 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | 117.015 ± 0.662 | 115.461 ± 0.911 | 113.214 ± 2.475 | 177.517 ± 1.361 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 1.338 ± 0.297 | 1.241 ± 0.006 | 1.257 ± 0.028 | 1.380 ± 0.016 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | 5.096 ± 0.033 | 5.025 ± 0.020 | 5.176 ± 0.188 | 6.091 ± 0.061 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | 23.189 ± 0.071 | 22.917 ± 0.106 | 23.263 ± 1.197 | 31.001 ± 0.188 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.350 ± 0.007 | 0.299 ± 0.002 | 0.299 ± 0.002 | 0.305 ± 0.003 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 117.060 ± 2.165 | 148.652 ± 21.004 | 181.425 ± 18.767 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.051 ± 0.046 | 5.511 ± 0.063 | 6.137 ± 0.068 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 22.928 ± 0.166 | 26.399 ± 0.828 | 31.786 ± 0.213 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 119.164 ± 4.613 | 129.121 ± 20.119 | 177.752 ± 1.588 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 5.003 ± 0.022 | 4.955 ± 0.016 | 6.040 ± 0.031 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 22.794 ± 0.107 | 23.328 ± 0.510 | 30.834 ± 0.071 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 4.525 ± 0.099 | 4.957 ± 0.227 | 4.596 ± 0.163 | 33.183 ± 0.907 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.014 ± 0.001 | 0.022 ± 0.000 | 0.014 ± 0.001 | 0.069 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.085 ± 0.001 | 0.101 ± 0.003 | 0.091 ± 0.006 | 0.555 ± 0.010 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.564 ± 0.006 | 0.655 ± 0.007 | 0.598 ± 0.017 | 4.205 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.210 ± 0.008 | 1.280 ± 0.024 | 1.239 ± 0.019 | 8.359 ± 0.253 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.288 ± 0.002 | 0.367 ± 0.003 | 0.343 ± 0.006 | 2.273 ± 0.168 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 25.266 ± 0.097 | 25.471 ± 0.124 | 33.561 ± 1.936 | 27.572 ± 1.353 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.377 ± 0.002 | 0.379 ± 0.002 | 0.400 ± 0.003 | 0.390 ± 0.006 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.171 ± 0.005 | 1.186 ± 0.003 | 1.365 ± 0.008 | 1.256 ± 0.018 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 5.010 ± 0.020 | 5.042 ± 0.024 | 6.270 ± 0.122 | 5.589 ± 0.221 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.071 ± 0.000 | 0.071 ± 0.001 | 0.071 ± 0.001 | 0.073 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 4.139 ± 0.318 | 4.120 ± 0.032 | 5.838 ± 1.406 | 5.091 ± 0.215 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 4.437 ± 0.503 | 4.032 ± 0.037 | 5.430 ± 0.733 | 5.064 ± 0.077 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 4.480 ± 0.112 | 4.451 ± 0.076 | 5.780 ± 0.180 | 5.471 ± 0.280 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.038 ± 0.001 | 0.043 ± 0.002 | 0.050 ± 0.001 | 0.056 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.042 ± 0.000 | 0.043 ± 0.000 | 0.051 ± 0.002 | 0.100 ± 0.010 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.063 ± 0.001 | 0.060 ± 0.002 | 0.062 ± 0.002 | 0.100 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.155 ± 0.003 | 0.164 ± 0.002 | 0.201 ± 0.008 | 0.218 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.152 ± 0.001 | 0.158 ± 0.004 | 0.203 ± 0.014 | 0.226 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.201 ± 0.002 | 0.186 ± 0.002 | 0.229 ± 0.010 | 0.317 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 0.797 ± 0.011 | 0.804 ± 0.012 | 1.058 ± 0.038 | 1.058 ± 0.048 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 0.789 ± 0.028 | 0.787 ± 0.004 | 1.031 ± 0.014 | 1.170 ± 0.551 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 0.926 ± 0.033 | 0.877 ± 0.013 | 1.151 ± 0.024 | 1.250 ± 0.117 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.019 ± 0.001 | 0.020 ± 0.003 | 0.027 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.025 ± 0.000 | 0.025 ± 0.001 | 0.024 ± 0.002 | 0.051 ± 0.006 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 105.664 ± 0.245 | 106.501 ± 0.270 | 106.516 ± 0.651 | 109.063 ± 1.440 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.212 ± 0.003 | 1.214 ± 0.004 | 1.215 ± 0.013 | 1.251 ± 0.014 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.820 ± 0.037 | 4.842 ± 0.020 | 4.843 ± 0.085 | 4.962 ± 0.075 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 21.638 ± 0.107 | 21.692 ± 0.051 | 21.901 ± 0.317 | 22.588 ± 0.407 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.279 ± 0.002 | 0.279 ± 0.002 | 0.276 ± 0.002 | 0.290 ± 0.017 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.042 ± 0.001 | 0.043 ± 0.001 | 0.040 ± 0.000 | 0.040 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.042 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.050 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.001 | 0.033 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.065 ± 0.002 | 0.073 ± 0.002 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.001 | 0.034 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.000 | 0.042 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.044 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.064 ± 0.001 | 0.068 ± 0.001 | 0.045 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.001 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.026 ± 0.001 | 0.037 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.033 ± 0.000 | 0.059 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.001 | 0.049 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.039 ± 0.000 | 0.075 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.001 | 0.050 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.000 | 0.057 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.033 ± 0.000 | 0.047 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.038 ± 0.000 | 0.057 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.000 | 0.046 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | 0.090 ± 0.001 | 0.021 ± 0.000 | 0.016 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.101 ± 0.002 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | 0.093 ± 0.002 | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.101 ± 0.001 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.083 ± 0.002 | 0.019 ± 0.000 | 0.015 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.050 ± 0.001 | 0.047 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.039 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.065 ± 0.001 | 0.055 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.040 ± 0.001 | 0.038 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.040 ± 0.001 | 0.045 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.050 ± 0.001 | 0.047 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.040 ± 0.000 | 0.036 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.066 ± 0.000 | 0.054 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.041 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.047 ± 0.000 | 0.044 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | 0.098 ± 0.002 | 0.021 ± 0.001 | 0.019 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.096 ± 0.003 | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | 0.113 ± 0.001 | 0.030 ± 0.000 | 0.027 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.095 ± 0.002 | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.097 ± 0.002 | 0.021 ± 0.001 | 0.019 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.017 ± 0.000 | 0.081 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.080 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.016 ± 0.000 | 0.080 ± 0.004 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.016 ± 0.000 | 0.080 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.016 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.015 ± 0.000 | 0.036 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.023 ± 0.000 | 0.050 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.015 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | 0.109 ± 0.001 | 0.039 ± 0.001 | 0.040 ± 0.000 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.083 ± 0.003 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | 0.157 ± 0.002 | 0.079 ± 0.000 | 0.079 ± 0.002 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.084 ± 0.002 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.090 ± 0.002 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.000 | 0.073 ± 0.001 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.080 ± 0.001 | 0.123 ± 0.002 | 0.070 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.015 ± 0.000 | 0.048 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.025 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.080 ± 0.001 | 0.102 ± 0.001 | 0.073 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | - | - |
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
| small | `svd` | f64 | 1 | `16x16` | - | 0.024 ± 0.000 | 0.025 ± 0.000 | 0.022 ± 0.000 | 0.025 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.071 ± 0.000 | 0.072 ± 0.000 | 0.069 ± 0.000 | 0.071 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 3.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`16x16`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 26.4x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`32x32`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_175502/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_175502`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_175502`.

- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_175502/cpu_ops_t4_20260923_175502.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_175502/cpu_ops_t4_20260923_175502.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 10.327 ± 0.301 | 10.014 ± 0.052 | 9.969 ± 0.073 | 5.287 ± 0.031 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.173 ± 0.015 | 0.162 ± 0.002 | 0.160 ± 0.001 | 0.145 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.534 ± 0.021 | 2.512 ± 0.019 | 2.570 ± 0.094 | 1.250 ± 0.127 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.637 ± 0.009 | 0.629 ± 0.003 | 0.645 ± 0.015 | 0.262 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.133 ± 0.003 | 0.128 ± 0.002 | 0.133 ± 0.006 | 0.273 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.036 ± 0.004 | 0.035 ± 0.000 | 0.035 ± 0.001 | 0.071 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.001 | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.750 ± 0.028 | 0.682 ± 0.004 | 0.787 ± 0.102 | 0.533 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.001 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.213 ± 0.007 | 0.175 ± 0.001 | 0.180 ± 0.020 | 0.212 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.052 ± 0.002 | 0.047 ± 0.000 | 0.045 ± 0.001 | 0.057 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 2.667 ± 0.036 | 2.663 ± 0.046 | 2.624 ± 0.109 | 1.232 ± 0.198 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.045 ± 0.001 | 0.045 ± 0.000 | 0.042 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.696 ± 0.024 | 0.662 ± 0.005 | 0.650 ± 0.011 | 0.330 ± 0.022 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.168 ± 0.002 | 0.167 ± 0.001 | 0.168 ± 0.002 | 0.177 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.257 ± 0.008 | 0.277 ± 0.028 | 1.008 ± 0.045 | 0.225 ± 0.031 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.008 ± 0.001 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.069 ± 0.005 | 0.090 ± 0.002 | 0.256 ± 0.025 | 0.093 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.001 | 0.026 ± 0.001 | 0.066 ± 0.003 | 0.050 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.003 | 0.034 ± 0.000 | 0.008 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.037 ± 0.001 | 0.045 ± 0.001 | 0.028 ± 0.001 | 0.095 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.039 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.009 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.088 ± 0.001 | 0.114 ± 0.004 | 0.479 ± 0.008 | 0.064 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.024 ± 0.001 | 0.030 ± 0.001 | 0.123 ± 0.001 | 0.047 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.008 ± 0.001 | 0.011 ± 0.000 | 0.035 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.888 ± 0.042 | 1.833 ± 0.022 | 1.716 ± 0.076 | 1.042 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.037 ± 0.005 | 0.034 ± 0.000 | 0.044 ± 0.001 | 0.037 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.473 ± 0.006 | 0.473 ± 0.009 | 0.460 ± 0.008 | 0.287 ± 0.029 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.132 ± 0.006 | 0.120 ± 0.001 | 0.134 ± 0.014 | 0.196 ± 0.036 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.142 ± 0.029 | 0.126 ± 0.001 | 0.120 ± 0.003 | 0.166 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.036 ± 0.001 | 0.034 ± 0.000 | 0.043 ± 0.002 | 0.045 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.306 ± 0.008 | 0.277 ± 0.002 | 0.271 ± 0.004 | 0.335 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.072 ± 0.008 | 0.072 ± 0.000 | 0.086 ± 0.008 | 0.086 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.021 ± 0.001 | 0.021 ± 0.000 | 0.032 ± 0.001 | 0.027 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.702 ± 0.009 | 0.690 ± 0.006 | 0.671 ± 0.005 | 0.369 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.190 ± 0.005 | 0.175 ± 0.001 | 0.175 ± 0.004 | 0.191 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.048 ± 0.001 | 0.048 ± 0.001 | 0.059 ± 0.004 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.580 ± 0.023 | 0.591 ± 0.011 | 0.318 ± 0.033 | 0.529 ± 0.048 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.016 ± 0.002 | 0.016 ± 0.000 | 0.026 ± 0.003 | 0.019 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.154 ± 0.002 | 0.157 ± 0.004 | 0.116 ± 0.015 | 0.148 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.045 ± 0.003 | 0.042 ± 0.001 | 0.041 ± 0.001 | 0.074 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.058 ± 0.001 | 0.055 ± 0.000 | 0.058 ± 0.002 | 0.080 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.003 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.020 ± 0.001 | 0.019 ± 0.000 | 0.031 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.097 ± 0.006 | 0.090 ± 0.003 | 0.076 ± 0.005 | 0.120 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.029 ± 0.002 | 0.027 ± 0.001 | 0.034 ± 0.001 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.001 | 0.011 ± 0.000 | 0.023 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.212 ± 0.005 | 0.212 ± 0.005 | 0.132 ± 0.009 | 0.171 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.002 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.060 ± 0.006 | 0.056 ± 0.001 | 0.048 ± 0.001 | 0.070 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 20.721 ± 0.167 | 20.541 ± 0.219 | 20.160 ± 0.092 | 10.518 ± 0.059 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.357 ± 0.029 | 0.324 ± 0.003 | 0.334 ± 0.014 | 0.189 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.163 ± 0.017 | 5.118 ± 0.028 | 5.065 ± 0.047 | 2.065 ± 0.634 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.384 ± 0.070 | 1.276 ± 0.008 | 1.274 ± 0.015 | 0.444 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.334 ± 0.042 | 0.318 ± 0.006 | 0.323 ± 0.009 | 1.103 ± 0.078 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.086 ± 0.002 | 0.082 ± 0.000 | 0.081 ± 0.003 | 0.282 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.001 | 0.025 ± 0.000 | 0.023 ± 0.000 | 0.072 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.399 ± 0.018 | 1.364 ± 0.014 | 1.387 ± 0.019 | 0.695 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.038 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.349 ± 0.004 | 0.350 ± 0.002 | 0.386 ± 0.018 | 0.518 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.100 ± 0.003 | 0.091 ± 0.001 | 0.088 ± 0.001 | 0.135 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 5.346 ± 0.187 | 5.197 ± 0.043 | 5.253 ± 0.094 | 2.576 ± 0.680 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.088 ± 0.002 | 0.088 ± 0.001 | 0.086 ± 0.001 | 0.096 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.411 ± 0.070 | 1.306 ± 0.010 | 1.302 ± 0.051 | 0.491 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.338 ± 0.008 | 0.330 ± 0.002 | 0.336 ± 0.007 | 0.318 ± 0.041 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.947 ± 0.039 | 0.842 ± 0.015 | 3.785 ± 0.112 | 0.600 ± 0.018 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.117 ± 0.012 | 0.038 ± 0.000 | 0.095 ± 0.002 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.316 ± 0.007 | 0.296 ± 0.006 | 1.035 ± 0.056 | 0.191 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.137 ± 0.006 | 0.088 ± 0.004 | 0.269 ± 0.006 | 0.061 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.162 ± 0.007 | 0.107 ± 0.002 | 0.040 ± 0.003 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.085 ± 0.002 | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.004 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.111 ± 0.005 | 0.039 ± 0.000 | 0.031 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.099 ± 0.012 | 0.021 ± 0.000 | 0.029 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.207 ± 0.002 | 0.149 ± 0.003 | 0.106 ± 0.013 | 0.260 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.087 ± 0.002 | 0.018 ± 0.000 | 0.030 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.141 ± 0.008 | 0.051 ± 0.001 | 0.041 ± 0.003 | 0.070 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.096 ± 0.005 | 0.024 ± 0.001 | 0.032 ± 0.001 | 0.020 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.573 ± 0.014 | 0.593 ± 0.014 | 3.168 ± 0.115 | 0.204 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.097 ± 0.001 | 0.029 ± 0.000 | 0.081 ± 0.002 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.221 ± 0.021 | 0.157 ± 0.003 | 0.827 ± 0.026 | 0.064 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.122 ± 0.002 | 0.061 ± 0.001 | 0.253 ± 0.013 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.074 ± 0.096 | 1.086 ± 0.019 | 0.516 ± 0.060 | 0.654 ± 0.025 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.128 ± 0.005 | 0.039 ± 0.000 | 0.050 ± 0.003 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.336 ± 0.004 | 0.310 ± 0.004 | 0.209 ± 0.016 | 0.193 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.161 ± 0.006 | 0.088 ± 0.002 | 0.080 ± 0.003 | 0.085 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.205 ± 0.003 | 0.154 ± 0.001 | 0.106 ± 0.006 | 0.106 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.109 ± 0.002 | 0.023 ± 0.000 | 0.038 ± 0.001 | 0.007 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.134 ± 0.002 | 0.055 ± 0.001 | 0.059 ± 0.004 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.125 ± 0.013 | 0.029 ± 0.000 | 0.042 ± 0.003 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.267 ± 0.006 | 0.211 ± 0.003 | 0.143 ± 0.014 | 0.157 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.111 ± 0.004 | 0.025 ± 0.000 | 0.041 ± 0.002 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.153 ± 0.003 | 0.070 ± 0.001 | 0.063 ± 0.003 | 0.045 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.124 ± 0.012 | 0.034 ± 0.000 | 0.048 ± 0.002 | 0.015 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.443 ± 0.007 | 0.424 ± 0.006 | 0.242 ± 0.006 | 0.227 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.140 ± 0.065 | 0.028 ± 0.000 | 0.043 ± 0.002 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.201 ± 0.010 | 0.117 ± 0.001 | 0.089 ± 0.007 | 0.083 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.148 ± 0.020 | 0.046 ± 0.000 | 0.055 ± 0.002 | 0.027 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 64.937 ± 4.813 | 66.424 ± 2.430 | 63.735 ± 0.248 | 66.336 ± 0.312 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.609 ± 0.005 | 0.629 ± 0.012 | 0.606 ± 0.007 | 0.656 ± 0.005 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.874 ± 0.086 | 2.725 ± 0.049 | 2.641 ± 0.064 | 2.797 ± 0.034 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 12.655 ± 0.484 | 12.495 ± 0.312 | 12.241 ± 0.257 | 12.489 ± 0.124 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.205 ± 0.003 | 0.206 ± 0.002 | 0.201 ± 0.002 | 0.212 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 71.233 ± 2.032 | 85.335 ± 20.792 | 84.465 ± 0.425 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 2.969 ± 0.027 | 3.048 ± 0.025 | 3.071 ± 0.031 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 13.510 ± 0.164 | 15.769 ± 1.132 | 15.136 ± 0.212 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 70.409 ± 2.052 | 71.667 ± 0.989 | 84.281 ± 0.250 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 2.938 ± 0.034 | 2.806 ± 0.016 | 3.025 ± 0.031 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 14.072 ± 1.119 | 13.177 ± 0.911 | 15.121 ± 0.146 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 27.330 ± 0.741 | 71.491 ± 20.395 | 34.819 ± 0.280 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.709 ± 0.017 | 1.343 ± 0.012 | 0.760 ± 0.009 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.313 ± 0.114 | 7.720 ± 0.277 | 5.370 ± 0.163 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 28.593 ± 1.309 | 30.614 ± 1.260 | 35.799 ± 1.600 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.712 ± 0.007 | 0.876 ± 0.294 | 0.805 ± 0.012 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 4.560 ± 0.345 | 4.251 ± 0.068 | 5.445 ± 0.082 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 6.665 ± 0.909 | 6.630 ± 0.294 | 5.160 ± 1.130 | 9.409 ± 0.122 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.028 ± 0.002 | 0.026 ± 0.001 | 0.018 ± 0.001 | 0.080 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.133 ± 0.006 | 0.127 ± 0.011 | 0.120 ± 0.005 | 0.191 ± 0.015 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 0.753 ± 0.025 | 0.717 ± 0.108 | 0.697 ± 0.037 | 1.254 ± 0.153 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.013 ± 0.001 | 0.011 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | 17.651 ± 2.406 | 17.902 ± 0.747 | 16.584 ± 2.564 | 26.451 ± 0.247 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.131 ± 0.002 | 0.078 ± 0.002 | 0.055 ± 0.001 | 0.191 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | 0.442 ± 0.022 | 0.392 ± 0.018 | 0.339 ± 0.012 | 0.545 ± 0.007 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | 2.272 ± 0.129 | 2.309 ± 0.245 | 2.018 ± 0.043 | 3.474 ± 0.069 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.103 ± 0.001 | 0.032 ± 0.001 | 0.024 ± 0.001 | 0.037 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 45.781 ± 0.747 | 51.149 ± 0.374 | 58.939 ± 1.235 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 1.842 ± 0.028 | 1.847 ± 0.020 | 1.804 ± 0.081 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 9.148 ± 0.640 | 9.190 ± 0.692 | 10.118 ± 0.215 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 46.771 ± 0.987 | 50.995 ± 10.517 | 63.035 ± 0.412 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 1.893 ± 0.030 | 1.785 ± 0.019 | 1.875 ± 0.009 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 9.351 ± 0.238 | 8.744 ± 0.796 | 10.308 ± 0.117 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | 5.750 ± 0.164 | 5.597 ± 0.201 | 5.398 ± 0.206 | 4.584 ± 0.120 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.157 ± 0.007 | 0.081 ± 0.002 | 0.077 ± 0.003 | 0.065 ± 0.002 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | 0.363 ± 0.015 | 0.271 ± 0.007 | 0.285 ± 0.020 | 0.217 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | 1.374 ± 0.074 | 1.165 ± 0.096 | 1.078 ± 0.053 | 1.004 ± 0.041 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.132 ± 0.003 | 0.041 ± 0.001 | 0.040 ± 0.002 | 0.022 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.021 ± 0.143 | 5.250 ± 0.064 | 5.215 ± 0.242 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.237 ± 0.003 | 0.318 ± 0.038 | 0.218 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.085 ± 0.142 | 1.056 ± 0.050 | 1.150 ± 0.015 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.081 ± 0.087 | 8.856 ± 0.115 | 4.841 ± 0.161 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.224 ± 0.005 | 0.447 ± 0.041 | 0.213 ± 0.003 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.010 ± 0.080 | 1.716 ± 0.018 | 1.102 ± 0.050 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | 123.745 ± 3.121 | 118.770 ± 8.662 | 117.020 ± 3.295 | 124.816 ± 0.426 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 1.335 ± 0.024 | 1.272 ± 0.017 | 1.236 ± 0.016 | 1.383 ± 0.025 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | 5.378 ± 0.077 | 5.206 ± 0.129 | 5.030 ± 0.086 | 5.225 ± 0.055 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | 23.975 ± 0.288 | 23.912 ± 1.279 | 23.370 ± 0.141 | 24.433 ± 0.215 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.361 ± 0.007 | 0.301 ± 0.005 | 0.295 ± 0.006 | 0.308 ± 0.004 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 133.188 ± 5.835 | 136.748 ± 18.239 | 127.161 ± 1.838 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.167 ± 0.093 | 6.067 ± 0.247 | 5.287 ± 0.031 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 25.994 ± 0.533 | 25.177 ± 0.265 | 25.389 ± 1.316 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 122.447 ± 12.308 | 113.170 ± 1.084 | 125.712 ± 1.869 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 5.099 ± 0.073 | 5.086 ± 0.132 | 5.233 ± 0.035 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 26.112 ± 1.561 | 22.682 ± 0.366 | 24.261 ± 0.142 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.774 ± 1.216 | 6.337 ± 2.831 | 4.818 ± 0.322 | 8.528 ± 0.046 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.014 ± 0.001 | 0.021 ± 0.001 | 0.014 ± 0.001 | 0.074 ± 0.003 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.085 ± 0.001 | 0.120 ± 0.003 | 0.096 ± 0.012 | 0.172 ± 0.030 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.566 ± 0.006 | 0.653 ± 0.027 | 0.616 ± 0.019 | 1.145 ± 0.017 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.232 ± 0.029 | 1.348 ± 0.139 | 1.258 ± 0.023 | 2.117 ± 0.038 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.323 ± 0.010 | 0.371 ± 0.015 | 0.340 ± 0.012 | 0.628 ± 0.014 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 25.608 ± 0.254 | 26.817 ± 1.177 | 27.935 ± 1.206 | 25.811 ± 0.365 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.395 ± 0.008 | 0.390 ± 0.008 | 0.398 ± 0.012 | 0.388 ± 0.003 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.245 ± 0.060 | 1.211 ± 0.015 | 1.233 ± 0.017 | 1.214 ± 0.183 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 5.251 ± 0.218 | 5.208 ± 0.064 | 5.569 ± 0.192 | 5.102 ± 0.054 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.073 ± 0.000 | 0.072 ± 0.001 | 0.081 ± 0.002 | 0.073 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 4.173 ± 0.162 | 4.577 ± 0.199 | 5.047 ± 0.428 | 4.404 ± 0.030 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 4.231 ± 0.391 | 4.385 ± 0.199 | 5.090 ± 0.577 | 4.407 ± 0.043 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 4.816 ± 0.468 | 4.830 ± 0.238 | 5.432 ± 0.384 | 4.788 ± 0.029 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.042 ± 0.004 | 0.046 ± 0.002 | 0.051 ± 0.002 | 0.055 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.043 ± 0.002 | 0.045 ± 0.000 | 0.050 ± 0.001 | 0.080 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.064 ± 0.003 | 0.058 ± 0.004 | 0.066 ± 0.007 | 0.095 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.158 ± 0.007 | 0.168 ± 0.003 | 0.203 ± 0.029 | 0.200 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.178 ± 0.025 | 0.166 ± 0.011 | 0.187 ± 0.011 | 0.223 ± 0.010 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.251 ± 0.016 | 0.199 ± 0.005 | 0.229 ± 0.021 | 0.261 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 0.838 ± 0.025 | 0.842 ± 0.052 | 0.943 ± 0.029 | 0.917 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 0.811 ± 0.024 | 0.817 ± 0.063 | 0.932 ± 0.034 | 0.960 ± 0.065 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 0.976 ± 0.017 | 0.915 ± 0.033 | 1.001 ± 0.020 | 1.073 ± 0.035 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.015 ± 0.001 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.024 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.026 ± 0.001 | 0.022 ± 0.000 | 0.025 ± 0.001 | 0.049 ± 0.004 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 107.877 ± 1.131 | 112.406 ± 2.720 | 109.051 ± 4.908 | 106.742 ± 0.899 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.273 ± 0.023 | 1.255 ± 0.026 | 1.215 ± 0.007 | 1.239 ± 0.013 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.995 ± 0.147 | 5.050 ± 0.049 | 4.833 ± 0.074 | 4.848 ± 0.049 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 22.094 ± 1.440 | 22.528 ± 0.239 | 22.936 ± 1.221 | 21.713 ± 0.144 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.288 ± 0.005 | 0.285 ± 0.002 | 0.278 ± 0.003 | 0.282 ± 0.004 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.013 ± 0.001 | 0.015 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.041 ± 0.000 | 0.042 ± 0.001 | 0.039 ± 0.000 | 0.040 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.038 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.037 ± 0.001 | 0.050 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.000 | 0.039 ± 0.004 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.065 ± 0.001 | 0.073 ± 0.001 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.035 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.028 ± 0.002 | 0.042 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.038 ± 0.001 | 0.044 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.001 | 0.030 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.064 ± 0.001 | 0.070 ± 0.002 | 0.044 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.026 ± 0.000 | 0.036 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.032 ± 0.000 | 0.140 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.000 | 0.130 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.039 ± 0.000 | 0.143 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.001 | 0.130 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.026 ± 0.000 | 0.134 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.032 ± 0.000 | 0.103 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.000 | 0.095 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.039 ± 0.000 | 0.104 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.000 | 0.097 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.030 ± 0.000 | 0.099 ± 0.001 | 0.005 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | 0.088 ± 0.008 | 0.020 ± 0.000 | 0.016 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.081 ± 0.009 | 0.016 ± 0.001 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | 0.091 ± 0.003 | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.080 ± 0.003 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.084 ± 0.016 | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.051 ± 0.001 | 0.077 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.041 ± 0.001 | 0.073 ± 0.004 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.061 ± 0.002 | 0.082 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.041 ± 0.001 | 0.073 ± 0.003 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.042 ± 0.001 | 0.078 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.051 ± 0.001 | 0.076 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.042 ± 0.001 | 0.078 ± 0.005 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.062 ± 0.001 | 0.086 ± 0.002 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.042 ± 0.001 | 0.073 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.047 ± 0.001 | 0.075 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | 0.102 ± 0.001 | 0.022 ± 0.000 | 0.020 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.103 ± 0.005 | 0.024 ± 0.002 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | 0.113 ± 0.001 | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.102 ± 0.002 | 0.021 ± 0.000 | 0.020 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.102 ± 0.004 | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.017 ± 0.000 | 0.086 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.017 ± 0.001 | 0.084 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.024 ± 0.000 | 0.090 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.017 ± 0.000 | 0.083 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.017 ± 0.000 | 0.083 ± 0.002 | 0.006 ± 0.003 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.017 ± 0.000 | 0.040 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.043 ± 0.007 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.023 ± 0.000 | 0.050 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.016 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.016 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | 0.131 ± 0.023 | 0.038 ± 0.001 | 0.039 ± 0.001 | 0.028 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.120 ± 0.020 | 0.015 ± 0.002 | 0.014 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | 0.158 ± 0.002 | 0.081 ± 0.002 | 0.084 ± 0.002 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.089 ± 0.002 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.092 ± 0.004 | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.001 | 0.076 ± 0.003 | 0.027 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.015 ± 0.001 | 0.048 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.082 ± 0.004 | 0.123 ± 0.002 | 0.076 ± 0.005 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.000 | 0.051 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.038 ± 0.001 | 0.056 ± 0.003 | 0.027 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.034 ± 0.004 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.083 ± 0.001 | 0.102 ± 0.002 | 0.075 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.014 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.019 ± 0.000 | 0.037 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.008 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.003 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.004 ± 0.001 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.007 ± 0.001 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.026 ± 0.001 | 0.026 ± 0.000 | 0.023 ± 0.000 | 0.026 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.072 ± 0.001 | 0.074 ± 0.001 | 0.070 ± 0.000 | 0.074 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.010 ± 0.001 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 3.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.827 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 26.7x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 36.1x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`16x16`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`32x32`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 22.9x faster than the slowest successful cell (`pytorch-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

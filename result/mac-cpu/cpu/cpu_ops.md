# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`
- Benchmark commit: `8595ce266fa34a87659e72c9b14195383343df95`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20260923_103427/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_103427`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_103427`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_103427/cpu_ops_t1_20260923_103427.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_103427/cpu_ops_t1_20260923_103427.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 10.879 ± 0.177 | 10.951 ± 0.072 | 9.894 ± 0.131 | 10.357 ± 0.165 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.172 ± 0.003 | 0.174 ± 0.003 | 0.159 ± 0.003 | 0.161 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.765 ± 0.049 | 2.734 ± 0.045 | 2.486 ± 0.050 | 2.540 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.687 ± 0.012 | 0.685 ± 0.017 | 0.623 ± 0.013 | 0.644 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.933 ± 0.011 | 0.944 ± 0.027 | 0.133 ± 0.005 | 0.264 ± 0.011 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.019 ± 0.001 | 0.019 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.233 ± 0.003 | 0.241 ± 0.006 | 0.034 ± 0.001 | 0.071 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.062 ± 0.001 | 0.064 ± 0.002 | 0.010 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.502 ± 0.040 | 1.543 ± 0.028 | 0.682 ± 0.015 | 0.818 ± 0.026 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.028 ± 0.001 | 0.028 ± 0.001 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.378 ± 0.008 | 0.386 ± 0.007 | 0.172 ± 0.004 | 0.207 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.098 ± 0.002 | 0.099 ± 0.002 | 0.044 ± 0.001 | 0.056 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 3.504 ± 0.046 | 3.467 ± 0.037 | 2.602 ± 0.052 | 2.728 ± 0.042 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.057 ± 0.001 | 0.058 ± 0.001 | 0.042 ± 0.001 | 0.047 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.859 ± 0.015 | 0.872 ± 0.010 | 0.645 ± 0.013 | 0.684 ± 0.020 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.219 ± 0.005 | 0.219 ± 0.003 | 0.162 ± 0.004 | 0.173 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.259 ± 0.026 | 0.313 ± 0.007 | 1.035 ± 0.015 | 0.349 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.066 ± 0.002 | 0.085 ± 0.003 | 0.252 ± 0.013 | 0.073 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.069 ± 0.002 | 0.048 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.001 | 0.033 ± 0.000 | 0.008 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.038 ± 0.002 | 0.047 ± 0.000 | 0.019 ± 0.001 | 0.047 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.008 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.085 ± 0.002 | 0.103 ± 0.004 | 0.489 ± 0.015 | 0.071 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.023 ± 0.001 | 0.030 ± 0.001 | 0.127 ± 0.002 | 0.048 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 1.833 ± 0.028 | 1.819 ± 0.025 | 1.844 ± 0.027 | 1.955 ± 0.028 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.033 ± 0.001 | 0.034 ± 0.000 | 0.031 ± 0.001 | 0.036 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.464 ± 0.013 | 0.469 ± 0.004 | 0.460 ± 0.010 | 0.501 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.123 ± 0.006 | 0.120 ± 0.001 | 0.117 ± 0.001 | 0.124 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.122 ± 0.002 | 0.124 ± 0.003 | 0.120 ± 0.002 | 0.161 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.034 ± 0.000 | 0.034 ± 0.001 | 0.031 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.272 ± 0.006 | 0.273 ± 0.002 | 0.274 ± 0.003 | 0.319 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.071 ± 0.001 | 0.071 ± 0.000 | 0.070 ± 0.001 | 0.084 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.021 ± 0.001 | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.676 ± 0.016 | 0.680 ± 0.010 | 0.679 ± 0.019 | 0.749 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.174 ± 0.001 | 0.176 ± 0.003 | 0.170 ± 0.003 | 0.188 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.046 ± 0.001 | 0.047 ± 0.001 | 0.044 ± 0.000 | 0.051 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.575 ± 0.014 | 2.445 ± 0.106 | 0.600 ± 0.028 | 0.782 ± 0.037 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.001 | 0.051 ± 0.001 | 0.014 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.148 ± 0.007 | 0.635 ± 0.008 | 0.153 ± 0.005 | 0.181 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.041 ± 0.001 | 0.166 ± 0.005 | 0.042 ± 0.002 | 0.049 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.056 ± 0.001 | 1.799 ± 0.053 | 0.060 ± 0.001 | 0.078 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.040 ± 0.001 | 0.005 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.464 ± 0.008 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.123 ± 0.004 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.089 ± 0.001 | 1.846 ± 0.048 | 0.096 ± 0.002 | 0.115 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.041 ± 0.001 | 0.006 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.026 ± 0.001 | 0.471 ± 0.021 | 0.028 ± 0.000 | 0.034 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.128 ± 0.005 | 0.011 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.200 ± 0.007 | 2.002 ± 0.047 | 0.214 ± 0.007 | 0.250 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.044 ± 0.001 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.055 ± 0.001 | 0.509 ± 0.018 | 0.057 ± 0.002 | 0.065 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.001 | 0.138 ± 0.003 | 0.018 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 21.475 ± 0.128 | 21.475 ± 0.229 | 20.066 ± 0.218 | 20.844 ± 0.172 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.342 ± 0.004 | 0.344 ± 0.003 | 0.319 ± 0.006 | 0.328 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.340 ± 0.034 | 5.427 ± 0.072 | 5.000 ± 0.128 | 5.163 ± 0.101 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.353 ± 0.013 | 1.348 ± 0.014 | 1.273 ± 0.020 | 1.299 ± 0.025 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.560 ± 0.024 | 1.584 ± 0.049 | 0.319 ± 0.003 | 1.034 ± 0.060 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.007 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.389 ± 0.007 | 0.392 ± 0.014 | 0.082 ± 0.002 | 0.260 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.102 ± 0.002 | 0.103 ± 0.004 | 0.022 ± 0.000 | 0.070 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.646 ± 0.043 | 2.641 ± 0.043 | 1.364 ± 0.022 | 2.041 ± 0.059 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.046 ± 0.001 | 0.047 ± 0.000 | 0.024 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.668 ± 0.006 | 0.656 ± 0.020 | 0.345 ± 0.009 | 0.515 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.170 ± 0.004 | 0.169 ± 0.003 | 0.088 ± 0.002 | 0.131 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 6.528 ± 0.089 | 6.572 ± 0.141 | 5.140 ± 0.106 | 5.702 ± 0.088 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.107 ± 0.002 | 0.108 ± 0.002 | 0.084 ± 0.002 | 0.094 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.629 ± 0.026 | 1.631 ± 0.025 | 1.281 ± 0.022 | 1.433 ± 0.024 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.415 ± 0.012 | 0.414 ± 0.008 | 0.327 ± 0.008 | 0.362 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.945 ± 0.050 | 0.909 ± 0.029 | 3.880 ± 0.053 | 0.889 ± 0.046 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.108 ± 0.002 | 0.038 ± 0.000 | 0.096 ± 0.002 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.286 ± 0.007 | 0.257 ± 0.005 | 0.966 ± 0.011 | 0.225 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.135 ± 0.002 | 0.092 ± 0.002 | 0.271 ± 0.006 | 0.057 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.157 ± 0.003 | 0.106 ± 0.002 | 0.039 ± 0.001 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.081 ± 0.003 | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.105 ± 0.004 | 0.038 ± 0.000 | 0.032 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.086 ± 0.003 | 0.021 ± 0.000 | 0.029 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.199 ± 0.004 | 0.150 ± 0.004 | 0.072 ± 0.002 | 0.173 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.003 | 0.018 ± 0.000 | 0.029 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.115 ± 0.002 | 0.051 ± 0.001 | 0.041 ± 0.001 | 0.048 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.089 ± 0.002 | 0.025 ± 0.000 | 0.031 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.549 ± 0.008 | 0.541 ± 0.009 | 3.228 ± 0.034 | 0.228 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.094 ± 0.001 | 0.031 ± 0.000 | 0.083 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.200 ± 0.002 | 0.156 ± 0.001 | 0.844 ± 0.008 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.117 ± 0.002 | 0.061 ± 0.001 | 0.232 ± 0.003 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 3.769 ± 0.058 | 4.592 ± 0.066 | 0.870 ± 0.051 | 0.896 ± 0.022 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.175 ± 0.005 | 0.105 ± 0.002 | 0.034 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 1.024 ± 0.012 | 1.173 ± 0.029 | 0.241 ± 0.005 | 0.218 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.339 ± 0.006 | 0.318 ± 0.010 | 0.075 ± 0.003 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 2.698 ± 0.021 | 3.431 ± 0.158 | 0.105 ± 0.003 | 0.103 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.165 ± 0.003 | 0.086 ± 0.002 | 0.022 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.750 ± 0.014 | 0.892 ± 0.029 | 0.042 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.277 ± 0.008 | 0.245 ± 0.006 | 0.026 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 2.773 ± 0.075 | 3.524 ± 0.081 | 0.154 ± 0.002 | 0.151 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.168 ± 0.003 | 0.089 ± 0.011 | 0.023 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.778 ± 0.032 | 0.908 ± 0.033 | 0.055 ± 0.001 | 0.044 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.279 ± 0.005 | 0.251 ± 0.007 | 0.030 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 2.963 ± 0.033 | 3.724 ± 0.063 | 0.316 ± 0.006 | 0.309 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.157 ± 0.004 | 0.092 ± 0.002 | 0.026 ± 0.001 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.826 ± 0.014 | 0.965 ± 0.017 | 0.096 ± 0.004 | 0.080 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.292 ± 0.002 | 0.267 ± 0.009 | 0.040 ± 0.001 | 0.026 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 62.070 ± 0.147 | 63.210 ± 0.253 | 63.497 ± 0.146 | 67.650 ± 0.415 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.605 ± 0.008 | 0.630 ± 0.003 | 0.609 ± 0.003 | 0.655 ± 0.031 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 2.582 ± 0.019 | 2.676 ± 0.019 | 2.649 ± 0.027 | 2.817 ± 0.017 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 11.647 ± 0.064 | 11.984 ± 0.090 | 12.072 ± 0.092 | 12.956 ± 0.130 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.200 ± 0.001 | 0.203 ± 0.002 | 0.199 ± 0.001 | 0.207 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 72.311 ± 0.242 | 80.435 ± 0.745 | 134.421 ± 0.594 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 2.970 ± 0.047 | 3.066 ± 0.056 | 3.914 ± 0.035 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 13.506 ± 0.055 | 14.412 ± 0.144 | 21.507 ± 0.117 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 71.564 ± 1.500 | 69.770 ± 0.370 | 133.219 ± 0.501 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 2.886 ± 0.018 | 2.817 ± 0.030 | 3.922 ± 0.041 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 13.141 ± 0.045 | 12.833 ± 0.081 | 21.273 ± 0.176 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 27.166 ± 0.820 | 58.127 ± 2.186 | 86.233 ± 0.400 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 0.659 ± 0.006 | 1.304 ± 0.018 | 1.743 ± 0.028 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 3.971 ± 0.031 | 8.383 ± 0.052 | 11.892 ± 0.091 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 28.206 ± 0.206 | 31.167 ± 0.682 | 87.195 ± 0.968 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 0.683 ± 0.006 | 0.768 ± 0.014 | 1.722 ± 0.006 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 4.004 ± 0.026 | 4.645 ± 0.070 | 11.839 ± 0.101 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 5.295 ± 0.214 | 5.473 ± 1.055 | 5.182 ± 0.982 | 33.586 ± 0.318 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.031 ± 0.001 | 0.028 ± 0.001 | 0.018 ± 0.001 | 0.078 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 0.117 ± 0.003 | 0.105 ± 0.003 | 0.103 ± 0.003 | 0.566 ± 0.007 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 0.709 ± 0.011 | 0.661 ± 0.024 | 0.631 ± 0.022 | 4.275 ± 0.038 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.012 ± 0.001 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | 16.611 ± 0.517 | 16.215 ± 0.574 | 15.355 ± 0.581 | 100.852 ± 0.717 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.126 ± 0.003 | 0.081 ± 0.001 | 0.060 ± 0.004 | 0.224 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | 0.397 ± 0.006 | 0.334 ± 0.005 | 0.291 ± 0.009 | 1.676 ± 0.010 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | 2.238 ± 0.030 | 2.124 ± 0.057 | 1.964 ± 0.089 | 12.786 ± 0.057 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.102 ± 0.001 | 0.033 ± 0.001 | 0.024 ± 0.001 | 0.036 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 49.295 ± 0.455 | 56.899 ± 0.535 | 134.856 ± 1.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 1.780 ± 0.013 | 1.918 ± 0.015 | 3.165 ± 0.011 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 8.562 ± 0.057 | 9.848 ± 0.071 | 18.879 ± 0.074 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 49.811 ± 0.713 | 54.337 ± 1.685 | 137.258 ± 0.596 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 1.834 ± 0.008 | 1.840 ± 0.021 | 3.147 ± 0.020 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 8.726 ± 0.019 | 9.321 ± 0.116 | 19.587 ± 0.031 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | 5.910 ± 0.094 | 5.605 ± 0.084 | 5.806 ± 0.101 | 5.148 ± 0.095 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.180 ± 0.001 | 0.099 ± 0.001 | 0.077 ± 0.003 | 0.062 ± 0.001 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | 0.352 ± 0.018 | 0.277 ± 0.005 | 0.256 ± 0.005 | 0.235 ± 0.009 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | 1.377 ± 0.117 | 1.168 ± 0.026 | 1.227 ± 0.029 | 1.031 ± 0.020 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.147 ± 0.001 | 0.054 ± 0.001 | 0.039 ± 0.001 | 0.022 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 5.213 ± 0.043 | 5.645 ± 0.096 | 5.153 ± 0.073 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.253 ± 0.003 | 0.294 ± 0.011 | 0.216 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.100 ± 0.019 | 1.184 ± 0.063 | 1.040 ± 0.034 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 5.050 ± 0.022 | 9.379 ± 0.086 | 4.789 ± 0.085 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.247 ± 0.002 | 0.385 ± 0.011 | 0.217 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 1.058 ± 0.025 | 1.881 ± 0.042 | 0.985 ± 0.017 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | 117.111 ± 1.589 | 116.026 ± 0.742 | 112.907 ± 1.182 | 176.897 ± 0.688 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 1.307 ± 0.006 | 1.251 ± 0.007 | 1.267 ± 0.029 | 1.379 ± 0.019 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | 5.060 ± 0.018 | 5.027 ± 0.015 | 4.976 ± 0.029 | 6.106 ± 0.075 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | 23.183 ± 0.099 | 22.904 ± 0.119 | 22.597 ± 0.142 | 30.786 ± 0.132 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.352 ± 0.006 | 0.301 ± 0.003 | 0.300 ± 0.003 | 0.305 ± 0.003 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 117.217 ± 0.418 | 133.723 ± 1.561 | 177.539 ± 0.348 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 5.046 ± 0.039 | 5.507 ± 0.049 | 6.085 ± 0.030 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 23.009 ± 0.088 | 25.876 ± 0.086 | 30.993 ± 0.205 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 116.310 ± 0.513 | 112.534 ± 0.699 | 176.879 ± 0.315 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 4.986 ± 0.027 | 4.976 ± 0.028 | 6.031 ± 0.039 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 22.916 ± 0.055 | 22.538 ± 0.084 | 30.751 ± 0.081 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 4.607 ± 0.895 | 5.538 ± 0.922 | 4.766 ± 0.218 | 33.090 ± 0.116 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.001 | 0.070 ± 0.001 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.086 ± 0.000 | 0.102 ± 0.002 | 0.096 ± 0.004 | 0.544 ± 0.008 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 0.566 ± 0.006 | 0.650 ± 0.024 | 0.605 ± 0.021 | 4.141 ± 0.027 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 1.233 ± 0.029 | 1.268 ± 0.019 | 1.248 ± 0.030 | 8.146 ± 0.031 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.315 ± 0.011 | 0.376 ± 0.006 | 0.337 ± 0.010 | 2.165 ± 0.037 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 25.379 ± 0.043 | 25.439 ± 0.104 | 30.696 ± 0.288 | 27.377 ± 0.248 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.379 ± 0.002 | 0.379 ± 0.002 | 0.402 ± 0.004 | 0.387 ± 0.003 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.185 ± 0.004 | 1.178 ± 0.004 | 1.342 ± 0.017 | 1.256 ± 0.012 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 5.052 ± 0.040 | 5.048 ± 0.044 | 6.197 ± 0.055 | 5.430 ± 0.048 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.071 ± 0.000 | 0.072 ± 0.000 | 0.070 ± 0.001 | 0.073 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 4.091 ± 0.077 | 4.467 ± 0.073 | 5.164 ± 0.097 | 5.048 ± 0.097 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 4.032 ± 0.094 | 4.539 ± 0.071 | 5.134 ± 0.066 | 5.078 ± 0.087 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 4.450 ± 0.111 | 4.962 ± 0.050 | 5.612 ± 0.132 | 5.439 ± 0.074 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.039 ± 0.001 | 0.060 ± 0.001 | 0.049 ± 0.001 | 0.054 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.041 ± 0.001 | 0.064 ± 0.002 | 0.051 ± 0.004 | 0.082 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.063 ± 0.001 | 0.079 ± 0.003 | 0.064 ± 0.004 | 0.095 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.155 ± 0.001 | 0.191 ± 0.003 | 0.194 ± 0.004 | 0.207 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.152 ± 0.004 | 0.195 ± 0.003 | 0.203 ± 0.014 | 0.223 ± 0.016 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.201 ± 0.001 | 0.248 ± 0.002 | 0.228 ± 0.009 | 0.266 ± 0.020 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 0.801 ± 0.011 | 0.915 ± 0.017 | 1.044 ± 0.029 | 1.021 ± 0.044 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 0.795 ± 0.017 | 0.914 ± 0.026 | 1.044 ± 0.036 | 1.033 ± 0.014 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 0.933 ± 0.009 | 1.054 ± 0.027 | 1.141 ± 0.017 | 1.181 ± 0.040 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.014 ± 0.000 | 0.026 ± 0.001 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.031 ± 0.000 | 0.019 ± 0.001 | 0.024 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.025 ± 0.001 | 0.039 ± 0.001 | 0.023 ± 0.001 | 0.048 ± 0.005 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 106.367 ± 0.666 | 106.412 ± 0.726 | 107.402 ± 0.593 | 108.408 ± 0.519 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.219 ± 0.009 | 1.223 ± 0.017 | 1.217 ± 0.010 | 1.239 ± 0.020 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 4.839 ± 0.024 | 4.823 ± 0.014 | 4.846 ± 0.014 | 4.901 ± 0.027 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 21.737 ± 0.160 | 21.730 ± 0.062 | 21.881 ± 0.106 | 22.174 ± 0.171 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.277 ± 0.001 | 0.280 ± 0.002 | 0.276 ± 0.002 | 0.282 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.042 ± 0.001 | 0.042 ± 0.000 | 0.040 ± 0.000 | 0.040 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.042 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.001 | 0.050 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.065 ± 0.002 | 0.074 ± 0.001 | 0.044 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.001 | 0.043 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.044 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.000 | 0.028 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.063 ± 0.000 | 0.068 ± 0.001 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.001 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.000 | 0.037 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.033 ± 0.000 | 0.060 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.039 ± 0.000 | 0.074 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.000 | 0.050 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.025 ± 0.001 | 0.058 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.033 ± 0.000 | 0.048 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.038 ± 0.000 | 0.056 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.024 ± 0.001 | 0.039 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | 0.090 ± 0.001 | 0.021 ± 0.000 | 0.016 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.102 ± 0.001 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | 0.091 ± 0.001 | 0.022 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.103 ± 0.001 | 0.015 ± 0.000 | 0.012 ± 0.001 | 0.004 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.083 ± 0.001 | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.050 ± 0.001 | 0.048 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.039 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.064 ± 0.001 | 0.055 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.039 ± 0.001 | 0.038 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.041 ± 0.000 | 0.045 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.051 ± 0.002 | 0.047 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.039 ± 0.002 | 0.037 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.066 ± 0.000 | 0.054 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.040 ± 0.000 | 0.038 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.047 ± 0.000 | 0.044 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | 0.121 ± 0.002 | 0.030 ± 0.001 | 0.019 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.118 ± 0.001 | 0.029 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | 0.134 ± 0.001 | 0.040 ± 0.000 | 0.027 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.118 ± 0.003 | 0.029 ± 0.000 | 0.018 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.118 ± 0.003 | 0.029 ± 0.001 | 0.019 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.024 ± 0.001 | 0.082 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.023 ± 0.000 | 0.081 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.032 ± 0.002 | 0.089 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.023 ± 0.000 | 0.082 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.023 ± 0.001 | 0.082 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.023 ± 0.000 | 0.038 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.022 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.032 ± 0.001 | 0.049 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.022 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.022 ± 0.000 | 0.038 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | 0.110 ± 0.001 | 0.039 ± 0.000 | 0.040 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.083 ± 0.002 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | 0.156 ± 0.002 | 0.080 ± 0.001 | 0.082 ± 0.003 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.085 ± 0.001 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.091 ± 0.003 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.040 ± 0.000 | 0.074 ± 0.001 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.014 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.080 ± 0.000 | 0.123 ± 0.002 | 0.075 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.015 ± 0.000 | 0.048 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.039 ± 0.000 | 0.056 ± 0.001 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.013 ± 0.000 | 0.031 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.080 ± 0.001 | 0.103 ± 0.002 | 0.076 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.024 ± 0.001 | 0.025 ± 0.000 | 0.022 ± 0.001 | 0.025 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.072 ± 0.000 | 0.072 ± 0.001 | 0.070 ± 0.000 | 0.072 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 30.0x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 32.3x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-trace`, 0.464 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 24.9x faster than the slowest successful cell (`tenferro-trace`, 0.464 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 26.3x faster than the slowest successful cell (`tenferro-trace`, 0.464 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 15.2x faster than the slowest successful cell (`tenferro-trace`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 1.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 19.2x faster than the slowest successful cell (`tenferro-trace`, 1.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 1.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 0.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`tenferro-trace`, 0.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 0.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-trace`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`tenferro-trace`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.105 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 3.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.844 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 33.2x faster than the slowest successful cell (`tenferro-trace`, 3.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 32.7x faster than the slowest successful cell (`tenferro-trace`, 3.431 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 28.8x faster than the slowest successful cell (`tenferro-trace`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 21.0x faster than the slowest successful cell (`tenferro-trace`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 28.9x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout),rhs=1`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 3.524 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 3.524 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 0.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.4x faster than the slowest successful cell (`tenferro-trace`, 0.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 3.724 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 3.724 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 0.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 0.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`16x16`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 26.8x faster than the slowest successful cell (`tenferro-eager`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`32x32`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 24.9x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20260923_105726/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260923_105726`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20260923_105726`.

- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

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


- CSV: `data/results/mac-cpu/cpu/einsum/20260923_105726/cpu_ops_t4_20260923_105726.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260923_105726/cpu_ops_t4_20260923_105726.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 10.955 ± 0.159 | 10.877 ± 0.255 | 9.800 ± 0.161 | 5.196 ± 2.482 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.173 ± 0.002 | 0.176 ± 0.002 | 0.158 ± 0.011 | 0.136 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.752 ± 0.060 | 2.730 ± 0.044 | 2.462 ± 0.039 | 1.160 ± 0.159 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.681 ± 0.014 | 0.687 ± 0.019 | 0.619 ± 0.005 | 0.260 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.947 ± 0.012 | 0.944 ± 0.023 | 0.129 ± 0.002 | 0.264 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.237 ± 0.002 | 0.239 ± 0.002 | 0.034 ± 0.009 | 0.071 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.065 ± 0.001 | 0.064 ± 0.001 | 0.010 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.534 ± 0.051 | 1.533 ± 0.024 | 0.678 ± 0.011 | 0.531 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.029 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.385 ± 0.005 | 0.386 ± 0.006 | 0.172 ± 0.001 | 0.211 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.003 | 0.099 ± 0.001 | 0.045 ± 0.001 | 0.057 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.480 ± 0.081 | 3.463 ± 0.055 | 2.534 ± 0.073 | 1.215 ± 0.110 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.057 ± 0.001 | 0.058 ± 0.001 | 0.042 ± 0.001 | 0.048 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.870 ± 0.013 | 0.879 ± 0.009 | 0.647 ± 0.031 | 0.321 ± 0.034 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.219 ± 0.003 | 0.220 ± 0.003 | 0.162 ± 0.003 | 0.175 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.217 ± 0.008 | 0.288 ± 0.044 | 1.008 ± 0.035 | 0.220 ± 0.025 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.002 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.065 ± 0.003 | 0.096 ± 0.002 | 0.259 ± 0.027 | 0.094 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.066 ± 0.004 | 0.049 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.030 ± 0.000 | 0.034 ± 0.000 | 0.008 ± 0.000 | 0.043 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.038 ± 0.001 | 0.046 ± 0.002 | 0.028 ± 0.002 | 0.091 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.008 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.086 ± 0.003 | 0.114 ± 0.003 | 0.478 ± 0.015 | 0.069 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.041 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.023 ± 0.001 | 0.029 ± 0.002 | 0.123 ± 0.003 | 0.048 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.034 ± 0.001 | 0.043 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.803 ± 0.049 | 1.792 ± 0.052 | 1.720 ± 0.035 | 1.038 ± 0.047 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.033 ± 0.000 | 0.034 ± 0.000 | 0.047 ± 0.002 | 0.037 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.475 ± 0.019 | 0.480 ± 0.008 | 0.463 ± 0.007 | 0.287 ± 0.014 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.119 ± 0.001 | 0.124 ± 0.003 | 0.125 ± 0.003 | 0.165 ± 0.028 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.125 ± 0.002 | 0.124 ± 0.001 | 0.120 ± 0.003 | 0.160 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.034 ± 0.001 | 0.035 ± 0.001 | 0.043 ± 0.003 | 0.045 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.274 ± 0.003 | 0.274 ± 0.002 | 0.268 ± 0.006 | 0.322 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.072 ± 0.002 | 0.072 ± 0.001 | 0.081 ± 0.002 | 0.086 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.033 ± 0.000 | 0.027 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.684 ± 0.011 | 0.683 ± 0.012 | 0.658 ± 0.017 | 0.370 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.175 ± 0.004 | 0.174 ± 0.002 | 0.176 ± 0.003 | 0.190 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.047 ± 0.000 | 0.057 ± 0.003 | 0.053 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.564 ± 0.012 | 2.397 ± 0.052 | 0.283 ± 0.013 | 0.506 ± 0.019 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.001 | 0.051 ± 0.001 | 0.026 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.156 ± 0.004 | 0.624 ± 0.009 | 0.101 ± 0.002 | 0.140 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.041 ± 0.001 | 0.167 ± 0.001 | 0.041 ± 0.001 | 0.075 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.056 ± 0.002 | 1.799 ± 0.020 | 0.059 ± 0.003 | 0.078 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.040 ± 0.001 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.457 ± 0.007 | 0.031 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.022 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.088 ± 0.002 | 1.863 ± 0.043 | 0.073 ± 0.002 | 0.118 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.041 ± 0.001 | 0.019 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.026 ± 0.000 | 0.482 ± 0.010 | 0.034 ± 0.001 | 0.035 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.127 ± 0.001 | 0.023 ± 0.001 | 0.010 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.202 ± 0.002 | 1.995 ± 0.022 | 0.129 ± 0.003 | 0.166 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.043 ± 0.001 | 0.021 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.055 ± 0.002 | 0.510 ± 0.007 | 0.048 ± 0.001 | 0.069 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.137 ± 0.001 | 0.029 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 21.537 ± 0.152 | 21.684 ± 0.287 | 20.051 ± 0.350 | 10.538 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.344 ± 0.004 | 0.346 ± 0.004 | 0.319 ± 0.004 | 0.182 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.483 ± 0.119 | 5.417 ± 0.038 | 5.051 ± 0.054 | 2.342 ± 0.383 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.364 ± 0.016 | 1.362 ± 0.019 | 1.257 ± 0.012 | 0.445 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.575 ± 0.022 | 1.589 ± 0.012 | 0.316 ± 0.004 | 1.038 ± 0.034 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.007 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.398 ± 0.003 | 0.398 ± 0.003 | 0.082 ± 0.001 | 0.265 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.101 ± 0.001 | 0.103 ± 0.001 | 0.022 ± 0.000 | 0.070 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.674 ± 0.029 | 2.670 ± 0.018 | 1.362 ± 0.025 | 0.666 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.046 ± 0.001 | 0.047 ± 0.001 | 0.024 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.668 ± 0.012 | 0.664 ± 0.019 | 0.344 ± 0.005 | 0.511 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.171 ± 0.002 | 0.170 ± 0.003 | 0.087 ± 0.002 | 0.132 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.465 ± 0.177 | 6.554 ± 0.170 | 5.022 ± 0.146 | 2.561 ± 0.335 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.106 ± 0.002 | 0.108 ± 0.003 | 0.084 ± 0.001 | 0.096 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.648 ± 0.037 | 1.640 ± 0.008 | 1.292 ± 0.018 | 0.492 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.414 ± 0.008 | 0.418 ± 0.003 | 0.324 ± 0.008 | 0.303 ± 0.029 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.930 ± 0.019 | 0.861 ± 0.022 | 3.789 ± 0.074 | 0.583 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.110 ± 0.001 | 0.039 ± 0.001 | 0.095 ± 0.002 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.333 ± 0.008 | 0.310 ± 0.007 | 0.969 ± 0.025 | 0.184 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.146 ± 0.003 | 0.099 ± 0.002 | 0.267 ± 0.003 | 0.061 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.162 ± 0.002 | 0.109 ± 0.001 | 0.039 ± 0.002 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.082 ± 0.002 | 0.017 ± 0.000 | 0.028 ± 0.001 | 0.004 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.104 ± 0.003 | 0.039 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.003 | 0.021 ± 0.000 | 0.029 ± 0.001 | 0.005 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.211 ± 0.003 | 0.157 ± 0.004 | 0.103 ± 0.005 | 0.257 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.082 ± 0.002 | 0.018 ± 0.000 | 0.029 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.118 ± 0.002 | 0.053 ± 0.001 | 0.040 ± 0.002 | 0.069 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.089 ± 0.002 | 0.025 ± 0.001 | 0.031 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.581 ± 0.014 | 0.579 ± 0.011 | 3.149 ± 0.045 | 0.201 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.095 ± 0.001 | 0.029 ± 0.003 | 0.081 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.209 ± 0.002 | 0.162 ± 0.002 | 0.832 ± 0.047 | 0.062 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.120 ± 0.002 | 0.063 ± 0.000 | 0.230 ± 0.003 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 3.624 ± 0.046 | 4.485 ± 0.075 | 0.496 ± 0.053 | 0.654 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.171 ± 0.002 | 0.105 ± 0.001 | 0.048 ± 0.004 | 0.020 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 1.038 ± 0.007 | 1.173 ± 0.019 | 0.200 ± 0.016 | 0.182 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.342 ± 0.008 | 0.322 ± 0.002 | 0.076 ± 0.002 | 0.086 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 2.720 ± 0.070 | 3.480 ± 0.040 | 0.104 ± 0.003 | 0.104 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.168 ± 0.002 | 0.087 ± 0.002 | 0.037 ± 0.001 | 0.007 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.757 ± 0.014 | 0.885 ± 0.010 | 0.055 ± 0.003 | 0.033 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.283 ± 0.002 | 0.246 ± 0.002 | 0.045 ± 0.002 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 2.840 ± 0.085 | 3.504 ± 0.058 | 0.132 ± 0.003 | 0.156 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.168 ± 0.003 | 0.089 ± 0.002 | 0.039 ± 0.001 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.789 ± 0.010 | 0.918 ± 0.008 | 0.065 ± 0.002 | 0.045 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.289 ± 0.005 | 0.255 ± 0.002 | 0.045 ± 0.003 | 0.015 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 2.940 ± 0.030 | 3.752 ± 0.025 | 0.243 ± 0.007 | 0.228 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.158 ± 0.004 | 0.093 ± 0.001 | 0.044 ± 0.001 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.855 ± 0.013 | 0.974 ± 0.011 | 0.088 ± 0.002 | 0.084 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.300 ± 0.005 | 0.271 ± 0.007 | 0.054 ± 0.001 | 0.028 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 63.378 ± 0.102 | 62.024 ± 0.164 | 63.038 ± 0.268 | 66.783 ± 0.159 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.620 ± 0.003 | 0.608 ± 0.006 | 0.608 ± 0.009 | 0.661 ± 0.008 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.694 ± 0.023 | 2.619 ± 0.020 | 2.618 ± 0.020 | 2.775 ± 0.062 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 12.103 ± 0.045 | 11.725 ± 0.052 | 11.862 ± 0.133 | 12.462 ± 0.086 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.202 ± 0.002 | 0.200 ± 0.001 | 0.198 ± 0.001 | 0.211 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 69.264 ± 0.141 | 79.046 ± 0.275 | 84.599 ± 0.809 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 2.921 ± 0.029 | 3.044 ± 0.048 | 3.052 ± 0.027 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 12.990 ± 0.066 | 14.081 ± 0.090 | 15.116 ± 0.065 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 67.779 ± 0.193 | 68.877 ± 0.223 | 85.003 ± 0.568 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 2.827 ± 0.022 | 2.798 ± 0.021 | 3.018 ± 0.028 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 12.643 ± 0.045 | 12.689 ± 0.048 | 15.089 ± 0.093 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 27.038 ± 0.164 | 53.674 ± 0.577 | 35.011 ± 0.388 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 0.739 ± 0.005 | 1.351 ± 0.016 | 0.753 ± 0.006 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 4.038 ± 0.032 | 7.557 ± 0.030 | 5.384 ± 0.127 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 27.529 ± 0.089 | 29.623 ± 0.156 | 35.767 ± 0.387 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 0.728 ± 0.007 | 0.848 ± 0.009 | 0.817 ± 0.012 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 3.920 ± 0.021 | 4.202 ± 0.027 | 5.453 ± 0.145 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 5.226 ± 0.181 | 4.820 ± 0.109 | 4.881 ± 1.158 | 8.649 ± 0.037 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.038 ± 0.001 | 0.036 ± 0.000 | 0.017 ± 0.001 | 0.080 ± 0.001 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.143 ± 0.002 | 0.132 ± 0.003 | 0.107 ± 0.005 | 0.184 ± 0.005 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 0.770 ± 0.011 | 0.676 ± 0.017 | 0.649 ± 0.015 | 1.193 ± 0.015 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.000 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | 15.830 ± 0.605 | 15.472 ± 0.159 | 14.808 ± 0.792 | 25.748 ± 0.064 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.132 ± 0.004 | 0.092 ± 0.003 | 0.057 ± 0.002 | 0.188 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | 0.434 ± 0.005 | 0.381 ± 0.004 | 0.324 ± 0.007 | 0.539 ± 0.012 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | 2.294 ± 0.071 | 2.064 ± 0.014 | 1.950 ± 0.038 | 3.455 ± 0.030 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.104 ± 0.001 | 0.035 ± 0.000 | 0.023 ± 0.001 | 0.038 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 46.479 ± 2.754 | 51.061 ± 0.249 | 59.802 ± 0.532 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 1.858 ± 0.009 | 1.843 ± 0.022 | 1.788 ± 0.037 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 8.015 ± 0.060 | 8.386 ± 0.058 | 9.571 ± 0.672 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 45.985 ± 0.190 | 48.811 ± 0.545 | 63.683 ± 0.317 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 1.849 ± 0.009 | 1.802 ± 0.029 | 1.877 ± 0.012 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 8.071 ± 0.044 | 8.117 ± 0.073 | 10.289 ± 0.128 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | 5.724 ± 0.086 | 5.365 ± 0.072 | 5.312 ± 0.089 | 4.641 ± 0.041 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.181 ± 0.002 | 0.099 ± 0.002 | 0.076 ± 0.003 | 0.061 ± 0.002 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | 0.390 ± 0.012 | 0.317 ± 0.004 | 0.251 ± 0.009 | 0.221 ± 0.011 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | 1.295 ± 0.072 | 1.152 ± 0.016 | 1.056 ± 0.040 | 0.954 ± 0.018 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.148 ± 0.003 | 0.055 ± 0.000 | 0.039 ± 0.001 | 0.022 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 5.015 ± 0.088 | 5.215 ± 0.132 | 5.047 ± 0.167 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.282 ± 0.004 | 0.293 ± 0.003 | 0.220 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.077 ± 0.027 | 1.038 ± 0.030 | 1.100 ± 0.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 4.879 ± 0.058 | 8.841 ± 0.113 | 4.746 ± 0.082 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.272 ± 0.005 | 0.406 ± 0.005 | 0.216 ± 0.002 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.045 ± 0.008 | 1.717 ± 0.014 | 1.096 ± 0.045 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | 115.694 ± 0.762 | 115.136 ± 0.446 | 111.671 ± 0.517 | 125.560 ± 0.341 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 1.298 ± 0.008 | 1.246 ± 0.007 | 1.248 ± 0.011 | 1.373 ± 0.015 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | 5.087 ± 0.016 | 5.006 ± 0.023 | 4.899 ± 0.017 | 5.241 ± 0.036 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | 23.072 ± 0.072 | 22.831 ± 0.090 | 22.823 ± 0.113 | 24.279 ± 0.105 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.351 ± 0.004 | 0.297 ± 0.001 | 0.297 ± 0.002 | 0.308 ± 0.003 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 116.195 ± 0.467 | 131.191 ± 0.857 | 127.753 ± 2.143 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.012 ± 0.030 | 5.488 ± 0.063 | 5.289 ± 0.063 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 22.902 ± 0.067 | 25.116 ± 0.496 | 24.413 ± 0.132 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 115.235 ± 0.458 | 111.634 ± 0.538 | 125.339 ± 0.828 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 5.011 ± 0.059 | 4.922 ± 0.039 | 5.224 ± 0.032 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 22.842 ± 0.124 | 22.431 ± 0.179 | 24.207 ± 0.071 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 4.804 ± 0.578 | 4.825 ± 0.440 | 4.631 ± 0.214 | 8.501 ± 0.040 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.014 ± 0.000 | 0.020 ± 0.002 | 0.014 ± 0.001 | 0.073 ± 0.001 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.085 ± 0.002 | 0.113 ± 0.001 | 0.089 ± 0.005 | 0.163 ± 0.002 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 0.564 ± 0.006 | 0.623 ± 0.012 | 0.609 ± 0.005 | 1.147 ± 0.014 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 1.208 ± 0.020 | 1.248 ± 0.010 | 1.234 ± 0.007 | 2.116 ± 0.016 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.287 ± 0.002 | 0.372 ± 0.010 | 0.334 ± 0.008 | 0.611 ± 0.012 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 25.406 ± 0.054 | 25.241 ± 0.052 | 27.371 ± 0.091 | 25.923 ± 0.061 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.377 ± 0.002 | 0.380 ± 0.002 | 0.395 ± 0.002 | 0.389 ± 0.004 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.186 ± 0.010 | 1.187 ± 0.004 | 1.236 ± 0.008 | 1.221 ± 0.013 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 5.012 ± 0.028 | 4.996 ± 0.025 | 5.325 ± 0.040 | 5.101 ± 0.026 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.071 ± 0.000 | 0.072 ± 0.000 | 0.081 ± 0.001 | 0.072 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 4.099 ± 0.048 | 4.381 ± 0.037 | 4.836 ± 0.073 | 4.448 ± 0.032 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 4.032 ± 0.035 | 4.407 ± 0.040 | 4.793 ± 0.212 | 4.453 ± 0.035 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 4.440 ± 0.053 | 4.851 ± 0.069 | 5.277 ± 0.213 | 4.815 ± 0.039 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.038 ± 0.001 | 0.060 ± 0.001 | 0.050 ± 0.002 | 0.056 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.041 ± 0.000 | 0.065 ± 0.001 | 0.051 ± 0.002 | 0.078 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.063 ± 0.001 | 0.083 ± 0.004 | 0.062 ± 0.002 | 0.094 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.150 ± 0.003 | 0.211 ± 0.002 | 0.179 ± 0.004 | 0.200 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.152 ± 0.002 | 0.212 ± 0.005 | 0.180 ± 0.006 | 0.220 ± 0.012 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.202 ± 0.005 | 0.270 ± 0.009 | 0.216 ± 0.010 | 0.263 ± 0.017 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 0.785 ± 0.012 | 0.911 ± 0.007 | 0.893 ± 0.020 | 0.920 ± 0.018 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 0.775 ± 0.021 | 0.905 ± 0.020 | 0.883 ± 0.014 | 0.927 ± 0.025 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 0.914 ± 0.032 | 1.044 ± 0.011 | 0.986 ± 0.013 | 1.060 ± 0.018 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.014 ± 0.000 | 0.027 ± 0.000 | 0.017 ± 0.001 | 0.019 ± 0.000 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.017 ± 0.000 | 0.031 ± 0.000 | 0.019 ± 0.000 | 0.024 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.026 ± 0.000 | 0.040 ± 0.001 | 0.024 ± 0.001 | 0.049 ± 0.005 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 106.506 ± 0.362 | 105.852 ± 0.584 | 106.899 ± 0.361 | 106.745 ± 0.509 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.219 ± 0.004 | 1.213 ± 0.004 | 1.212 ± 0.005 | 1.241 ± 0.015 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 4.875 ± 0.031 | 4.824 ± 0.024 | 4.791 ± 0.028 | 4.843 ± 0.016 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 21.703 ± 0.079 | 21.627 ± 0.055 | 22.153 ± 0.074 | 21.689 ± 0.107 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.277 ± 0.001 | 0.281 ± 0.003 | 0.276 ± 0.002 | 0.284 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.041 ± 0.000 | 0.043 ± 0.001 | 0.040 ± 0.001 | 0.041 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.036 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.041 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.041 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.001 | 0.050 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.067 ± 0.002 | 0.074 ± 0.001 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.031 ± 0.001 | 0.042 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.001 | 0.045 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.028 ± 0.001 | 0.003 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.064 ± 0.000 | 0.069 ± 0.002 | 0.044 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.000 | 0.029 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.036 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.033 ± 0.000 | 0.140 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.001 | 0.129 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.042 ± 0.000 | 0.150 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.000 | 0.131 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.133 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.033 ± 0.001 | 0.100 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.001 | 0.095 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.039 ± 0.001 | 0.109 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.095 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.031 ± 0.000 | 0.099 ± 0.002 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | 0.089 ± 0.001 | 0.021 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.077 ± 0.001 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | 0.091 ± 0.001 | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.077 ± 0.001 | 0.015 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.083 ± 0.001 | 0.019 ± 0.000 | 0.015 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.051 ± 0.002 | 0.076 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.040 ± 0.001 | 0.069 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.065 ± 0.005 | 0.084 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.040 ± 0.001 | 0.070 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.000 | 0.075 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.052 ± 0.001 | 0.082 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.040 ± 0.001 | 0.076 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.066 ± 0.001 | 0.088 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.040 ± 0.000 | 0.076 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.048 ± 0.000 | 0.078 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | 0.120 ± 0.001 | 0.030 ± 0.001 | 0.019 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.118 ± 0.002 | 0.029 ± 0.001 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | 0.135 ± 0.002 | 0.040 ± 0.000 | 0.027 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.119 ± 0.001 | 0.029 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.121 ± 0.000 | 0.030 ± 0.000 | 0.019 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.024 ± 0.000 | 0.081 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.024 ± 0.001 | 0.080 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.034 ± 0.000 | 0.089 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.023 ± 0.001 | 0.079 ± 0.002 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.024 ± 0.000 | 0.083 ± 0.003 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.023 ± 0.001 | 0.038 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.023 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.032 ± 0.001 | 0.047 ± 0.005 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.023 ± 0.000 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.023 ± 0.001 | 0.037 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | 0.111 ± 0.001 | 0.039 ± 0.000 | 0.040 ± 0.001 | 0.029 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.083 ± 0.001 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | 0.156 ± 0.001 | 0.079 ± 0.001 | 0.084 ± 0.003 | 0.078 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.086 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.090 ± 0.001 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.040 ± 0.000 | 0.074 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.001 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.081 ± 0.000 | 0.123 ± 0.002 | 0.076 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.015 ± 0.000 | 0.048 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.001 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.039 ± 0.001 | 0.056 ± 0.000 | 0.029 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.013 ± 0.000 | 0.030 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.080 ± 0.002 | 0.102 ± 0.001 | 0.077 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.018 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.000 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.001 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.012 ± 0.001 | 0.005 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.012 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.012 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.024 ± 0.000 | 0.026 ± 0.001 | 0.023 ± 0.001 | 0.025 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.071 ± 0.000 | 0.073 ± 0.000 | 0.070 ± 0.000 | 0.072 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 30.7x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 31.9x faster than the slowest successful cell (`tenferro-trace`, 1.799 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 0.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 0.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 26.0x faster than the slowest successful cell (`tenferro-trace`, 0.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-trace`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 1.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 1.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 1.863 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-trace`, 0.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 0.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 18.5x faster than the slowest successful cell (`tenferro-trace`, 0.482 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-trace`, 1.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`tenferro-trace`, 1.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 0.510 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-eager`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `jax-cpu` is 17.9x faster than the slowest successful cell (`tenferro-eager`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 3.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.832 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 33.5x faster than the slowest successful cell (`tenferro-trace`, 3.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 33.6x faster than the slowest successful cell (`tenferro-trace`, 3.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 25.2x faster than the slowest successful cell (`tenferro-eager`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 26.7x faster than the slowest successful cell (`tenferro-trace`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 16.2x faster than the slowest successful cell (`tenferro-trace`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 30.2x faster than the slowest successful cell (`tenferro-eager`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 22.5x faster than the slowest successful cell (`tenferro-trace`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 29.7x faster than the slowest successful cell (`tenferro-eager`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 20.4x faster than the slowest successful cell (`tenferro-trace`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-trace`, 3.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout),rhs=1`): `pytorch-cpu` is 15.4x faster than the slowest successful cell (`tenferro-trace`, 3.752 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout),rhs=1`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 0.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 28.3x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 35.8x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 25.6x faster than the slowest successful cell (`pytorch-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`16x16`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`32x32`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.9x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 24.7x faster than the slowest successful cell (`tenferro-eager`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`tenferro-eager`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

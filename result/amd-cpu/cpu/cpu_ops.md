# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`
- Benchmark commit: `2cd0fd6445dcd9aad98990f76c17e33e773c4a7d`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261008_044146`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261008_044146`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_t1_20261008_044146.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_t1_20261008_044146.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 20.298 ± 0.027 | 20.286 ± 0.029 | 21.490 ± 0.084 | 26.954 ± 0.323 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.305 ± 0.001 | 0.306 ± 0.000 | 0.324 ± 0.001 | 0.444 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.069 ± 0.009 | 5.082 ± 0.014 | 5.422 ± 0.025 | 7.246 ± 0.150 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.240 ± 0.003 | 1.245 ± 0.002 | 1.332 ± 0.003 | 1.800 ± 0.068 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.619 ± 0.002 | 0.621 ± 0.002 | 0.399 ± 0.001 | 0.365 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.160 ± 0.000 | 0.162 ± 0.000 | 0.105 ± 0.001 | 0.094 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.045 ± 0.000 | 0.047 ± 0.000 | 0.032 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.239 ± 0.011 | 2.244 ± 0.003 | 2.079 ± 0.006 | 2.142 ± 0.076 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.043 ± 0.000 | 0.039 ± 0.000 | 0.041 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.563 ± 0.001 | 0.566 ± 0.001 | 0.524 ± 0.001 | 0.542 ± 0.022 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.145 ± 0.000 | 0.148 ± 0.000 | 0.136 ± 0.000 | 0.137 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 6.230 ± 0.009 | 6.235 ± 0.009 | 6.238 ± 0.022 | 7.372 ± 0.167 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.103 ± 0.000 | 0.104 ± 0.000 | 0.104 ± 0.000 | 0.119 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.565 ± 0.002 | 1.565 ± 0.001 | 1.567 ± 0.004 | 1.864 ± 0.061 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.393 ± 0.001 | 0.395 ± 0.001 | 0.396 ± 0.001 | 0.460 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.751 ± 0.004 | 0.894 ± 0.004 | 0.827 ± 0.010 | 0.921 ± 0.061 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.120 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.193 ± 0.002 | 0.234 ± 0.002 | 0.209 ± 0.002 | 0.174 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.053 ± 0.000 | 0.062 ± 0.000 | 0.059 ± 0.001 | 0.125 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.100 ± 0.000 | 0.106 ± 0.000 | 0.034 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.116 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.029 ± 0.000 | 0.034 ± 0.000 | 0.017 ± 0.000 | 0.120 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.000 | 0.118 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.119 ± 0.000 | 0.130 ± 0.001 | 0.128 ± 0.000 | 0.125 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.128 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.034 ± 0.000 | 0.039 ± 0.000 | 0.041 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.205 ± 0.001 | 0.244 ± 0.002 | 0.191 ± 0.003 | 0.166 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.055 ± 0.000 | 0.065 ± 0.000 | 0.055 ± 0.001 | 0.125 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.121 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 4.298 ± 0.011 | 4.281 ± 0.004 | 4.628 ± 0.027 | 7.303 ± 0.446 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.073 ± 0.000 | 0.074 ± 0.000 | 0.076 ± 0.001 | 0.111 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 1.075 ± 0.002 | 1.080 ± 0.002 | 1.164 ± 0.009 | 1.742 ± 0.091 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.271 ± 0.002 | 0.274 ± 0.001 | 0.292 ± 0.002 | 0.430 ± 0.014 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.204 ± 0.001 | 0.207 ± 0.000 | 0.204 ± 0.001 | 0.268 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.056 ± 0.000 | 0.058 ± 0.000 | 0.055 ± 0.000 | 0.073 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.579 ± 0.002 | 0.582 ± 0.001 | 0.540 ± 0.002 | 0.777 ± 0.057 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.150 ± 0.000 | 0.152 ± 0.000 | 0.138 ± 0.001 | 0.202 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.042 ± 0.000 | 0.044 ± 0.000 | 0.038 ± 0.000 | 0.053 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 1.378 ± 0.004 | 1.380 ± 0.004 | 1.367 ± 0.006 | 2.270 ± 0.057 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.030 ± 0.000 | 0.026 ± 0.000 | 0.041 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.351 ± 0.001 | 0.352 ± 0.001 | 0.347 ± 0.000 | 0.554 ± 0.014 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.093 ± 0.000 | 0.094 ± 0.000 | 0.091 ± 0.000 | 0.149 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.341 ± 0.003 | 1.400 ± 0.006 | 1.496 ± 0.006 | 2.610 ± 0.182 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.030 ± 0.000 | 0.031 ± 0.000 | 0.038 ± 0.000 | 0.046 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.336 ± 0.001 | 0.341 ± 0.001 | 0.386 ± 0.003 | 0.614 ± 0.042 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.091 ± 0.001 | 0.089 ± 0.000 | 0.107 ± 0.001 | 0.158 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.086 ± 0.000 | 0.064 ± 0.000 | 0.083 ± 0.000 | 0.250 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.033 ± 0.000 | 0.067 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.143 ± 0.000 | 0.107 ± 0.000 | 0.121 ± 0.002 | 0.392 ± 0.027 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.042 ± 0.000 | 0.034 ± 0.000 | 0.042 ± 0.000 | 0.103 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.447 ± 0.001 | 0.426 ± 0.001 | 0.431 ± 0.003 | 0.810 ± 0.037 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.116 ± 0.000 | 0.110 ± 0.000 | 0.119 ± 0.001 | 0.208 ± 0.016 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.036 ± 0.000 | 0.036 ± 0.000 | 0.041 ± 0.000 | 0.058 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 43.728 ± 0.040 | 43.712 ± 0.043 | 47.372 ± 0.106 | 47.504 ± 0.217 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.671 ± 0.002 | 0.675 ± 0.001 | 0.727 ± 0.001 | 0.814 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 10.920 ± 0.023 | 10.932 ± 0.017 | 11.910 ± 0.038 | 12.250 ± 0.156 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 2.715 ± 0.009 | 2.721 ± 0.003 | 2.949 ± 0.016 | 3.172 ± 0.107 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.361 ± 0.007 | 1.360 ± 0.004 | 1.113 ± 0.004 | 1.365 ± 0.053 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.031 ± 0.000 | 0.025 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.346 ± 0.001 | 0.347 ± 0.001 | 0.285 ± 0.001 | 0.346 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.092 ± 0.000 | 0.094 ± 0.000 | 0.077 ± 0.000 | 0.089 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 4.732 ± 0.008 | 4.740 ± 0.008 | 4.230 ± 0.011 | 4.213 ± 0.126 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.000 | 0.084 ± 0.000 | 0.073 ± 0.000 | 0.074 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 1.188 ± 0.002 | 1.192 ± 0.002 | 1.066 ± 0.001 | 1.054 ± 0.033 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.302 ± 0.001 | 0.305 ± 0.000 | 0.271 ± 0.001 | 0.266 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 15.250 ± 0.041 | 15.302 ± 0.010 | 14.465 ± 0.036 | 13.177 ± 0.178 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.242 ± 0.000 | 0.246 ± 0.000 | 0.231 ± 0.000 | 0.224 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 3.817 ± 0.007 | 3.836 ± 0.007 | 3.624 ± 0.009 | 3.412 ± 0.094 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.961 ± 0.002 | 0.960 ± 0.001 | 0.909 ± 0.002 | 0.882 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.884 ± 0.026 | 7.845 ± 0.046 | 2.440 ± 0.252 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.000 | 0.209 ± 0.002 | 0.038 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.772 ± 0.004 | 2.033 ± 0.016 | 0.568 ± 0.035 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.214 ± 0.001 | 0.577 ± 0.004 | 0.140 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.325 ± 0.001 | 0.150 ± 0.001 | 0.023 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.083 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.107 ± 0.000 | 0.099 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.086 ± 0.001 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.415 ± 0.001 | 0.442 ± 0.002 | 0.439 ± 0.020 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.088 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.129 ± 0.000 | 0.173 ± 0.001 | 0.115 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.057 ± 0.000 | 0.104 ± 0.001 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.861 ± 0.003 | 5.565 ± 0.029 | 0.795 ± 0.027 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.171 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.238 ± 0.001 | 1.523 ± 0.014 | 0.207 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.083 ± 0.000 | 0.432 ± 0.001 | 0.055 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.397 ± 0.006 | 3.029 ± 0.187 | 4.313 ± 0.365 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.000 | 0.107 ± 0.001 | 0.076 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.644 ± 0.002 | 0.821 ± 0.029 | 0.970 ± 0.033 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.184 ± 0.001 | 0.251 ± 0.005 | 0.259 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.254 ± 0.001 | 0.179 ± 0.001 | 0.531 ± 0.034 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.058 ± 0.000 | 0.021 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.094 ± 0.000 | 0.089 ± 0.000 | 0.137 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.000 | 0.065 ± 0.000 | 0.042 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.336 ± 0.001 | 0.287 ± 0.002 | 0.810 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.062 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.113 ± 0.000 | 0.116 ± 0.001 | 0.210 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.000 | 0.073 ± 0.000 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.089 ± 0.003 | 1.017 ± 0.011 | 1.506 ± 0.036 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.056 ± 0.000 | 0.072 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.301 ± 0.000 | 0.304 ± 0.003 | 0.376 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.104 ± 0.000 | 0.119 ± 0.000 | 0.103 ± 0.004 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 282.723 ± 0.915 | 282.184 ± 0.431 | 291.704 ± 0.378 | 198.438 ± 2.902 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 1.394 ± 0.012 | 1.399 ± 0.010 | 1.470 ± 0.005 | 1.464 ± 0.068 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 7.577 ± 0.018 | 7.561 ± 0.029 | 7.730 ± 0.020 | 6.760 ± 0.194 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 45.400 ± 0.105 | 45.351 ± 0.181 | 46.627 ± 0.124 | 34.732 ± 0.177 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.306 ± 0.000 | 0.308 ± 0.001 | 0.316 ± 0.001 | 0.374 ± 0.018 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 348.894 ± 0.537 | 493.351 ± 1.190 | 323.311 ± 0.574 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 8.925 ± 0.019 | 11.077 ± 0.037 | 8.384 ± 0.027 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 55.310 ± 0.214 | 74.532 ± 0.165 | 51.286 ± 0.247 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 350.559 ± 0.476 | 356.213 ± 0.625 | 320.925 ± 0.413 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 8.878 ± 0.023 | 8.978 ± 0.048 | 8.350 ± 0.064 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 55.276 ± 0.129 | 56.729 ± 0.082 | 51.124 ± 0.305 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 282.367 ± 0.360 | 609.271 ± 2.501 | 234.359 ± 2.491 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 5.342 ± 0.017 | 10.825 ± 0.076 | 4.599 ± 0.016 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 38.496 ± 0.209 | 82.831 ± 0.187 | 32.653 ± 0.091 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 276.074 ± 0.653 | 326.508 ± 5.939 | 235.357 ± 1.248 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 5.465 ± 0.006 | 5.832 ± 0.023 | 4.504 ± 0.022 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 38.204 ± 0.065 | 43.382 ± 0.117 | 31.662 ± 0.072 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 63.075 ± 0.306 | 59.155 ± 0.532 | 67.647 ± 0.602 | 61.157 ± 0.195 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.161 ± 0.001 | 0.155 ± 0.000 | 0.149 ± 0.003 | 0.157 ± 0.012 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 1.022 ± 0.004 | 1.002 ± 0.003 | 1.035 ± 0.007 | 1.073 ± 0.038 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 7.788 ± 0.035 | 7.788 ± 0.066 | 8.791 ± 0.034 | 8.054 ± 0.289 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.037 ± 0.000 | 0.032 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 179.406 ± 0.802 | 201.239 ± 1.248 | 184.856 ± 1.855 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.490 ± 0.001 | 0.476 ± 0.002 | 0.429 ± 0.023 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.043 ± 0.007 | 3.059 ± 0.010 | 3.219 ± 0.103 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 23.303 ± 0.112 | 25.201 ± 0.071 | 23.433 ± 0.317 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.099 ± 0.001 | 0.098 ± 0.001 | 0.064 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 365.127 ± 0.494 | 418.926 ± 1.604 | 301.561 ± 1.265 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 7.699 ± 0.038 | 8.272 ± 0.018 | 6.457 ± 0.017 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 57.267 ± 0.091 | 66.265 ± 0.747 | 42.961 ± 0.350 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 360.289 ± 0.408 | 407.364 ± 2.039 | 331.027 ± 0.619 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 7.658 ± 0.014 | 8.287 ± 0.025 | 6.462 ± 0.041 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 54.552 ± 0.101 | 68.759 ± 0.876 | 47.262 ± 0.587 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 28.162 ± 0.092 | 40.933 ± 0.322 | 22.921 ± 0.420 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.165 ± 0.001 | 0.204 ± 0.003 | 0.166 ± 0.007 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.741 ± 0.004 | 0.881 ± 0.008 | 0.701 ± 0.024 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.275 ± 0.015 | 5.399 ± 0.040 | 4.665 ± 0.160 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.071 ± 0.000 | 0.089 ± 0.001 | 0.051 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 27.206 ± 0.092 | 36.963 ± 0.129 | 22.241 ± 0.089 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.720 ± 0.006 | 1.068 ± 0.017 | 0.610 ± 0.008 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.274 ± 0.066 | 5.398 ± 0.025 | 4.051 ± 0.020 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 26.804 ± 0.083 | 65.111 ± 0.487 | 22.017 ± 0.050 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.705 ± 0.006 | 1.564 ± 0.010 | 0.623 ± 0.003 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.216 ± 0.026 | 9.538 ± 0.026 | 4.087 ± 0.015 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 681.870 ± 2.592 | 748.617 ± 2.088 | 611.029 ± 3.765 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.977 ± 0.018 | 3.188 ± 0.016 | 2.925 ± 0.029 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 16.134 ± 0.092 | 16.864 ± 0.039 | 14.794 ± 0.174 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 108.161 ± 1.700 | 126.310 ± 0.256 | 100.161 ± 0.449 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.642 ± 0.002 | 0.700 ± 0.010 | 0.605 ± 0.017 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 676.216 ± 3.145 | 908.650 ± 4.960 | 607.557 ± 3.247 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 16.134 ± 0.091 | 20.226 ± 0.068 | 14.537 ± 0.023 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 107.400 ± 0.993 | 148.129 ± 0.664 | 99.509 ± 0.661 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 661.978 ± 15.387 | 684.864 ± 6.642 | 607.308 ± 1.994 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 16.094 ± 0.039 | 16.906 ± 0.064 | 14.569 ± 0.050 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 106.149 ± 2.119 | 125.661 ± 0.489 | 99.615 ± 0.461 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 58.006 ± 0.103 | 58.975 ± 0.130 | 61.822 ± 0.287 | 60.039 ± 0.149 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.159 ± 0.001 | 0.151 ± 0.001 | 0.166 ± 0.006 | 0.143 ± 0.017 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 1.029 ± 0.004 | 0.998 ± 0.002 | 1.084 ± 0.012 | 1.120 ± 0.042 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 7.684 ± 0.044 | 7.631 ± 0.061 | 8.030 ± 0.022 | 7.916 ± 0.287 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 14.489 ± 0.046 | 14.623 ± 0.057 | 15.236 ± 0.056 | 15.123 ± 0.152 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.759 ± 0.008 | 3.879 ± 0.022 | 4.073 ± 0.028 | 4.233 ± 0.221 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 130.893 ± 0.353 | 131.362 ± 1.052 | 159.844 ± 1.398 | 98.438 ± 1.121 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.535 ± 0.001 | 0.537 ± 0.001 | 0.650 ± 0.003 | 0.609 ± 0.036 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 3.139 ± 0.022 | 3.145 ± 0.012 | 3.834 ± 0.019 | 2.771 ± 0.121 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 22.256 ± 0.113 | 22.569 ± 0.179 | 26.527 ± 0.070 | 17.895 ± 0.442 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.096 ± 0.000 | 0.097 ± 0.000 | 0.105 ± 0.001 | 0.097 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 25.679 ± 0.037 | 25.851 ± 0.146 | 39.283 ± 0.336 | 22.535 ± 0.380 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 27.069 ± 0.079 | 27.195 ± 0.059 | 41.276 ± 0.470 | 23.113 ± 0.640 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 30.498 ± 0.112 | 30.523 ± 0.096 | 45.731 ± 0.374 | 25.874 ± 0.321 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.109 ± 0.000 | 0.113 ± 0.001 | 0.138 ± 0.000 | 0.137 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.147 ± 0.003 | 0.153 ± 0.004 | 0.181 ± 0.002 | 0.394 ± 0.038 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.247 ± 0.004 | 0.257 ± 0.003 | 0.289 ± 0.003 | 0.580 ± 0.296 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.612 ± 0.002 | 0.630 ± 0.004 | 0.767 ± 0.004 | 0.636 ± 0.039 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.728 ± 0.003 | 0.744 ± 0.001 | 0.887 ± 0.007 | 1.146 ± 0.275 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 1.021 ± 0.009 | 1.040 ± 0.004 | 1.237 ± 0.011 | 1.398 ± 0.173 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 3.932 ± 0.018 | 3.999 ± 0.033 | 5.073 ± 0.023 | 4.447 ± 0.390 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 4.356 ± 0.009 | 4.410 ± 0.030 | 5.517 ± 0.034 | 4.606 ± 0.363 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 5.352 ± 0.023 | 5.406 ± 0.008 | 6.674 ± 0.030 | 5.503 ± 0.231 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.033 ± 0.000 | 0.036 ± 0.000 | 0.041 ± 0.000 | 0.040 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.044 ± 0.001 | 0.048 ± 0.000 | 0.054 ± 0.001 | 0.109 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.075 ± 0.000 | 0.079 ± 0.002 | 0.092 ± 0.002 | 0.127 ± 0.031 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 561.278 ± 2.950 | 558.175 ± 4.959 | 620.126 ± 14.833 | 460.655 ± 0.877 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 2.658 ± 0.013 | 2.677 ± 0.009 | 2.818 ± 0.009 | 2.718 ± 0.174 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 14.186 ± 0.056 | 14.232 ± 0.054 | 15.282 ± 0.025 | 12.953 ± 0.137 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 91.733 ± 0.686 | 91.706 ± 0.674 | 99.442 ± 0.293 | 82.792 ± 0.279 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.576 ± 0.002 | 0.578 ± 0.003 | 0.599 ± 0.002 | 0.615 ± 0.031 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.035 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.078 ± 0.000 | 0.079 ± 0.000 | 0.084 ± 0.000 | 0.103 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.127 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.115 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.131 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.117 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.117 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.066 ± 0.000 | 0.124 ± 0.001 | 0.032 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.046 ± 0.000 | 0.097 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.126 ± 0.001 | 0.197 ± 0.003 | 0.102 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.047 ± 0.000 | 0.098 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.052 ± 0.000 | 0.105 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.068 ± 0.000 | 0.107 ± 0.001 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.046 ± 0.000 | 0.083 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.125 ± 0.000 | 0.171 ± 0.004 | 0.103 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.048 ± 0.000 | 0.086 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.053 ± 0.000 | 0.090 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.055 ± 0.000 | 0.175 ± 0.004 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.049 ± 0.000 | 0.158 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.079 ± 0.000 | 0.216 ± 0.004 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.049 ± 0.000 | 0.157 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.051 ± 0.000 | 0.161 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.056 ± 0.000 | 0.138 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.049 ± 0.000 | 0.127 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.083 ± 0.000 | 0.166 ± 0.004 | 0.031 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.050 ± 0.000 | 0.127 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.051 ± 0.000 | 0.130 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.036 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.032 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.046 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.034 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.033 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.091 ± 0.001 | 0.124 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.081 ± 0.000 | 0.112 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.119 ± 0.001 | 0.156 ± 0.002 | 0.039 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.082 ± 0.000 | 0.110 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.084 ± 0.001 | 0.114 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.094 ± 0.000 | 0.131 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.083 ± 0.000 | 0.118 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.125 ± 0.000 | 0.161 ± 0.004 | 0.039 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.084 ± 0.000 | 0.117 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.086 ± 0.000 | 0.119 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.055 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.051 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.063 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.052 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.053 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.266 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.028 ± 0.000 | 0.259 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.038 ± 0.000 | 0.277 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.029 ± 0.000 | 0.257 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.029 ± 0.000 | 0.259 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.117 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.028 ± 0.000 | 0.112 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.037 ± 0.000 | 0.132 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.028 ± 0.000 | 0.111 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.029 ± 0.000 | 0.114 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.000 | 0.085 ± 0.000 | 0.057 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.038 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.173 ± 0.001 | 0.203 ± 0.001 | 0.173 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.042 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.053 ± 0.000 | 0.020 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.070 ± 0.000 | 0.210 ± 0.005 | 0.053 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.027 ± 0.000 | 0.147 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.174 ± 0.000 | 0.346 ± 0.009 | 0.158 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.030 ± 0.000 | 0.151 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.041 ± 0.000 | 0.167 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.069 ± 0.000 | 0.144 ± 0.001 | 0.054 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.026 ± 0.000 | 0.092 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.173 ± 0.001 | 0.269 ± 0.005 | 0.160 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.030 ± 0.000 | 0.096 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.041 ± 0.000 | 0.107 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.029 ± 0.002 | 0.029 ± 0.000 | 0.028 ± 0.001 | 0.029 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.025 ± 0.000 | 0.027 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.000 | 0.017 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.047 ± 0.000 | 0.050 ± 0.000 | 0.052 ± 0.000 | 0.057 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.151 ± 0.000 | 0.153 ± 0.001 | 0.165 ± 0.001 | 0.171 ± 0.007 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.020 ± 0.000 | 0.023 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-trace`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 25.5x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 34.2x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 34.6x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

### Collection recovery

Completed einsum measurements retained from the initial collection; CPU ops were recollected after the input-restoration fix. Component commits and original metadata are recorded in `run.yaml` and `einsum_run.yaml`. Exact commands: `data/results/amd-cpu/cpu/refresh/20261008_044136/commands_remaining.sh`.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261008_053424/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261008_053424`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261008_053424`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

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

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261008_053424/cpu_ops_t4_20261008_053424.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261008_053424/cpu_ops_t4_20261008_053424.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 20.313 ± 0.022 | 20.315 ± 0.036 | 21.490 ± 0.079 | 14.120 ± 0.200 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.305 ± 0.001 | 0.306 ± 0.001 | 0.324 ± 0.001 | 0.311 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.083 ± 0.014 | 5.072 ± 0.005 | 5.366 ± 0.015 | 3.230 ± 0.679 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.242 ± 0.002 | 1.244 ± 0.002 | 1.333 ± 0.009 | 0.653 ± 0.118 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.606 ± 0.002 | 0.603 ± 0.002 | 0.403 ± 0.003 | 0.355 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.157 ± 0.000 | 0.157 ± 0.000 | 0.105 ± 0.001 | 0.094 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.046 ± 0.000 | 0.031 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.210 ± 0.005 | 2.207 ± 0.007 | 2.055 ± 0.004 | 1.520 ± 0.245 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.042 ± 0.000 | 0.039 ± 0.000 | 0.041 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.557 ± 0.001 | 0.558 ± 0.002 | 0.517 ± 0.002 | 0.545 ± 0.018 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.144 ± 0.000 | 0.145 ± 0.000 | 0.134 ± 0.000 | 0.139 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 6.230 ± 0.012 | 6.234 ± 0.020 | 6.230 ± 0.011 | 3.408 ± 0.391 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.102 ± 0.000 | 0.104 ± 0.000 | 0.103 ± 0.000 | 0.120 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.564 ± 0.005 | 1.559 ± 0.005 | 1.571 ± 0.009 | 0.834 ± 0.161 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.394 ± 0.001 | 0.394 ± 0.000 | 0.397 ± 0.001 | 0.460 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.238 ± 0.001 | 0.308 ± 0.009 | 0.436 ± 0.025 | 1.376 ± 0.293 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.001 | 0.118 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.063 ± 0.001 | 0.108 ± 0.001 | 0.114 ± 0.020 | 0.262 ± 0.052 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.037 ± 0.000 | 0.035 ± 0.005 | 0.127 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.066 ± 0.001 | 0.075 ± 0.001 | 0.034 ± 0.000 | 0.123 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.116 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.022 ± 0.000 | 0.027 ± 0.000 | 0.017 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.117 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.071 ± 0.000 | 0.087 ± 0.000 | 0.063 ± 0.007 | 0.216 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.023 ± 0.000 | 0.030 ± 0.000 | 0.041 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.118 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.093 ± 0.001 | 0.137 ± 0.001 | 0.105 ± 0.010 | 0.417 ± 0.048 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.028 ± 0.000 | 0.044 ± 0.000 | 0.035 ± 0.006 | 0.127 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.000 | 0.119 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 4.248 ± 0.017 | 4.247 ± 0.009 | 4.382 ± 0.033 | 3.485 ± 0.124 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.072 ± 0.000 | 0.074 ± 0.001 | 0.075 ± 0.000 | 0.112 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 1.062 ± 0.005 | 1.067 ± 0.002 | 1.107 ± 0.006 | 0.913 ± 0.063 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.270 ± 0.001 | 0.270 ± 0.001 | 0.296 ± 0.035 | 0.292 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.197 ± 0.001 | 0.197 ± 0.001 | 0.167 ± 0.001 | 0.266 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.054 ± 0.000 | 0.056 ± 0.000 | 0.046 ± 0.000 | 0.070 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.574 ± 0.002 | 0.600 ± 0.002 | 0.496 ± 0.003 | 0.823 ± 0.048 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.148 ± 0.000 | 0.150 ± 0.001 | 0.129 ± 0.001 | 0.204 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.042 ± 0.000 | 0.046 ± 0.000 | 0.037 ± 0.000 | 0.055 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 1.376 ± 0.006 | 1.346 ± 0.002 | 1.287 ± 0.007 | 1.062 ± 0.104 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.042 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.340 ± 0.001 | 0.346 ± 0.003 | 0.333 ± 0.001 | 0.565 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.090 ± 0.000 | 0.091 ± 0.000 | 0.088 ± 0.000 | 0.152 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.343 ± 0.005 | 1.398 ± 0.008 | 0.691 ± 0.006 | 1.835 ± 0.144 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.030 ± 0.000 | 0.030 ± 0.000 | 0.029 ± 0.001 | 0.050 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.339 ± 0.000 | 0.345 ± 0.002 | 0.195 ± 0.008 | 0.525 ± 0.040 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.089 ± 0.000 | 0.089 ± 0.000 | 0.072 ± 0.004 | 0.158 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.086 ± 0.000 | 0.064 ± 0.000 | 0.070 ± 0.009 | 0.256 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.029 ± 0.000 | 0.069 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.022 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.147 ± 0.000 | 0.105 ± 0.000 | 0.098 ± 0.003 | 0.389 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.043 ± 0.000 | 0.034 ± 0.000 | 0.036 ± 0.000 | 0.101 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.466 ± 0.002 | 0.429 ± 0.001 | 0.266 ± 0.001 | 0.806 ± 0.039 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.021 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.118 ± 0.000 | 0.110 ± 0.000 | 0.090 ± 0.003 | 0.212 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.033 ± 0.000 | 0.059 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 43.820 ± 0.031 | 43.840 ± 0.127 | 47.367 ± 0.130 | 23.766 ± 10.673 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.675 ± 0.002 | 0.682 ± 0.003 | 0.727 ± 0.005 | 0.340 ± 0.052 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 10.935 ± 0.008 | 11.014 ± 0.024 | 11.909 ± 0.023 | 5.096 ± 0.626 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 2.731 ± 0.004 | 2.735 ± 0.004 | 2.969 ± 0.008 | 0.919 ± 0.158 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.307 ± 0.003 | 1.317 ± 0.003 | 1.052 ± 0.002 | 1.322 ± 0.032 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.030 ± 0.000 | 0.024 ± 0.000 | 0.029 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.332 ± 0.000 | 0.336 ± 0.001 | 0.269 ± 0.001 | 0.336 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.000 | 0.092 ± 0.000 | 0.074 ± 0.000 | 0.087 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 4.699 ± 0.011 | 4.722 ± 0.011 | 4.209 ± 0.014 | 1.735 ± 0.245 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.000 | 0.083 ± 0.000 | 0.073 ± 0.000 | 0.073 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 1.181 ± 0.001 | 1.189 ± 0.001 | 1.057 ± 0.002 | 1.065 ± 0.036 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.000 | 0.303 ± 0.000 | 0.269 ± 0.001 | 0.270 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 15.292 ± 0.019 | 15.292 ± 0.055 | 14.464 ± 0.027 | 5.079 ± 1.443 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.243 ± 0.001 | 0.248 ± 0.001 | 0.231 ± 0.001 | 0.213 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 3.819 ± 0.005 | 3.850 ± 0.008 | 3.636 ± 0.013 | 1.069 ± 0.098 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.959 ± 0.003 | 0.961 ± 0.003 | 0.911 ± 0.003 | 0.587 ± 0.067 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.122 ± 0.013 | 7.643 ± 0.046 | 3.150 ± 0.553 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.000 | 0.204 ± 0.001 | 0.045 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.370 ± 0.002 | 1.941 ± 0.004 | 0.721 ± 0.086 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.127 ± 0.000 | 0.549 ± 0.002 | 0.157 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.231 ± 0.001 | 0.151 ± 0.000 | 0.024 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.084 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.087 ± 0.000 | 0.101 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.087 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.280 ± 0.004 | 0.215 ± 0.004 | 0.665 ± 0.058 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.088 ± 0.001 | 0.016 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.099 ± 0.000 | 0.174 ± 0.002 | 0.170 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.001 | 0.105 ± 0.001 | 0.047 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.476 ± 0.002 | 5.518 ± 0.034 | 1.353 ± 0.208 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.172 ± 0.001 | 0.024 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.153 ± 0.001 | 1.443 ± 0.010 | 0.270 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.067 ± 0.001 | 0.429 ± 0.001 | 0.073 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.240 ± 0.021 | 1.341 ± 0.020 | 3.669 ± 0.434 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.000 | 0.099 ± 0.001 | 0.080 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.619 ± 0.002 | 0.441 ± 0.006 | 0.927 ± 0.162 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.182 ± 0.000 | 0.211 ± 0.002 | 0.268 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.223 ± 0.001 | 0.164 ± 0.001 | 0.534 ± 0.018 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.061 ± 0.001 | 0.021 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.087 ± 0.000 | 0.088 ± 0.002 | 0.136 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.000 | 0.066 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.303 ± 0.001 | 0.258 ± 0.002 | 0.817 ± 0.038 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.063 ± 0.000 | 0.023 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.106 ± 0.001 | 0.110 ± 0.000 | 0.210 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.058 ± 0.000 | 0.072 ± 0.001 | 0.060 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.045 ± 0.002 | 0.666 ± 0.010 | 1.695 ± 0.237 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.057 ± 0.000 | 0.072 ± 0.001 | 0.036 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.292 ± 0.001 | 0.267 ± 0.002 | 0.384 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.102 ± 0.000 | 0.113 ± 0.001 | 0.106 ± 0.005 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 90.456 ± 0.288 | 89.951 ± 0.274 | 93.152 ± 2.203 | 112.289 ± 6.261 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 0.801 ± 0.004 | 0.801 ± 0.010 | 0.879 ± 0.014 | 1.481 ± 0.068 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 3.670 ± 0.005 | 3.665 ± 0.035 | 3.859 ± 0.150 | 5.964 ± 0.499 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 16.318 ± 0.081 | 16.293 ± 0.107 | 17.235 ± 0.358 | 24.346 ± 0.306 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.217 ± 0.000 | 0.219 ± 0.001 | 0.238 ± 0.002 | 0.383 ± 0.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 110.595 ± 1.190 | 158.889 ± 0.821 | 142.499 ± 2.814 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 4.346 ± 0.042 | 4.765 ± 0.018 | 7.224 ± 0.444 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 19.898 ± 0.698 | 25.879 ± 0.099 | 30.225 ± 0.888 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 110.256 ± 0.489 | 112.010 ± 1.240 | 141.547 ± 1.556 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 4.343 ± 0.053 | 4.113 ± 0.017 | 6.640 ± 0.531 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 20.180 ± 0.389 | 19.726 ± 0.056 | 29.945 ± 1.244 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 81.038 ± 0.600 | 180.950 ± 0.557 | 70.229 ± 3.164 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 1.911 ± 0.025 | 4.312 ± 0.052 | 1.852 ± 0.325 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 12.071 ± 1.972 | 30.860 ± 0.381 | 10.272 ± 1.694 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 79.189 ± 0.214 | 92.926 ± 0.252 | 68.551 ± 1.687 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 2.167 ± 0.024 | 2.125 ± 0.016 | 1.753 ± 0.087 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 11.894 ± 0.095 | 14.338 ± 0.081 | 10.084 ± 0.540 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 16.928 ± 0.137 | 16.718 ± 0.361 | 21.150 ± 0.199 | 16.143 ± 0.184 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.075 ± 0.001 | 0.072 ± 0.000 | 0.076 ± 0.010 | 0.091 ± 0.013 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.364 ± 0.003 | 0.354 ± 0.008 | 0.367 ± 0.037 | 0.405 ± 0.029 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 2.594 ± 0.055 | 2.464 ± 0.057 | 4.255 ± 0.293 | 2.637 ± 0.221 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.030 ± 0.000 | 0.027 ± 0.000 | 0.022 ± 0.002 | 0.030 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 52.029 ± 0.484 | 57.790 ± 0.072 | 48.312 ± 1.548 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.231 ± 0.007 | 0.216 ± 0.008 | 0.184 ± 0.023 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.091 ± 0.006 | 1.033 ± 0.031 | 1.057 ± 0.219 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 7.223 ± 0.064 | 9.424 ± 0.188 | 6.895 ± 0.331 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.000 | 0.080 ± 0.002 | 0.065 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 113.904 ± 0.650 | 131.511 ± 0.602 | 112.847 ± 0.720 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 3.322 ± 0.030 | 3.344 ± 0.594 | 3.794 ± 0.308 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 22.380 ± 0.604 | 25.576 ± 0.157 | 19.457 ± 1.904 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 111.495 ± 0.286 | 126.004 ± 0.331 | 114.857 ± 2.341 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 3.191 ± 0.047 | 3.534 ± 0.046 | 3.846 ± 0.345 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 20.052 ± 0.347 | 25.700 ± 0.527 | 20.776 ± 2.378 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 9.094 ± 0.106 | 11.418 ± 2.027 | 9.588 ± 0.765 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.168 ± 0.002 | 0.206 ± 0.001 | 0.155 ± 0.001 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.381 ± 0.015 | 0.459 ± 0.019 | 0.627 ± 0.075 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.524 ± 0.016 | 2.492 ± 0.190 | 2.535 ± 0.261 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.072 ± 0.000 | 0.089 ± 0.001 | 0.053 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 8.244 ± 1.917 | 11.528 ± 0.113 | 9.718 ± 0.888 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.350 ± 0.002 | 0.624 ± 0.019 | 0.446 ± 0.013 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.491 ± 0.079 | 3.028 ± 0.098 | 2.864 ± 0.761 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 7.920 ± 0.100 | 18.763 ± 0.164 | 9.293 ± 0.231 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.347 ± 0.009 | 0.701 ± 0.018 | 0.450 ± 0.020 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.437 ± 0.006 | 3.944 ± 0.017 | 2.570 ± 0.497 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 265.586 ± 4.853 | 306.521 ± 4.105 | 361.237 ± 6.286 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.255 ± 0.047 | 2.591 ± 0.029 | 2.707 ± 0.078 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 10.753 ± 0.041 | 12.119 ± 0.061 | 12.847 ± 0.799 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 51.535 ± 0.476 | 69.670 ± 2.524 | 75.418 ± 1.222 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.603 ± 0.004 | 0.659 ± 0.006 | 0.635 ± 0.065 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 265.149 ± 2.585 | 344.704 ± 1.889 | 354.826 ± 6.893 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 10.761 ± 0.090 | 12.931 ± 0.063 | 12.295 ± 0.613 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 50.354 ± 1.528 | 68.240 ± 0.589 | 75.044 ± 1.311 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 266.303 ± 5.420 | 280.895 ± 4.921 | 360.760 ± 4.876 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 10.735 ± 0.024 | 12.043 ± 0.043 | 12.622 ± 1.063 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 50.639 ± 0.336 | 68.901 ± 1.123 | 75.148 ± 0.982 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 14.772 ± 0.045 | 16.309 ± 0.058 | 17.586 ± 0.085 | 15.938 ± 0.087 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.062 ± 0.001 | 0.066 ± 0.000 | 0.081 ± 0.008 | 0.077 ± 0.014 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.319 ± 0.005 | 0.352 ± 0.002 | 0.386 ± 0.029 | 0.385 ± 0.057 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 2.175 ± 0.012 | 2.361 ± 0.045 | 3.003 ± 0.376 | 2.449 ± 0.267 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 3.668 ± 0.007 | 3.885 ± 0.043 | 4.721 ± 0.356 | 4.407 ± 0.155 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.073 ± 0.004 | 1.241 ± 0.020 | 1.562 ± 0.368 | 1.493 ± 0.235 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 45.073 ± 0.465 | 44.836 ± 0.214 | 52.003 ± 0.354 | 59.527 ± 7.731 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.420 ± 0.002 | 0.423 ± 0.006 | 0.412 ± 0.003 | 0.683 ± 0.075 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.570 ± 0.007 | 1.580 ± 0.038 | 1.734 ± 0.027 | 3.460 ± 0.349 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 8.984 ± 0.029 | 9.086 ± 0.018 | 10.307 ± 0.169 | 12.304 ± 0.660 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.107 ± 0.000 | 0.108 ± 0.001 | 0.114 ± 0.002 | 0.099 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 6.948 ± 0.091 | 7.140 ± 0.025 | 9.780 ± 1.760 | 10.195 ± 0.803 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 7.937 ± 0.055 | 8.122 ± 0.021 | 10.693 ± 1.943 | 9.494 ± 0.319 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 8.703 ± 0.058 | 9.029 ± 0.100 | 12.180 ± 1.530 | 10.494 ± 1.066 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.109 ± 0.000 | 0.115 ± 0.000 | 0.141 ± 0.001 | 0.138 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.132 ± 0.002 | 0.138 ± 0.001 | 0.164 ± 0.001 | 0.217 ± 0.044 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.165 ± 0.004 | 0.174 ± 0.002 | 0.206 ± 0.002 | 0.369 ± 0.155 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.250 ± 0.001 | 0.268 ± 0.002 | 0.353 ± 0.032 | 0.555 ± 0.137 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.312 ± 0.004 | 0.332 ± 0.011 | 0.379 ± 0.070 | 0.805 ± 0.313 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.409 ± 0.002 | 0.428 ± 0.007 | 0.486 ± 0.008 | 0.822 ± 0.232 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 1.263 ± 0.004 | 1.327 ± 0.012 | 1.737 ± 0.199 | 2.228 ± 0.469 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 1.461 ± 0.033 | 1.585 ± 0.005 | 1.822 ± 0.019 | 2.445 ± 0.209 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 1.725 ± 0.014 | 1.853 ± 0.013 | 2.179 ± 0.044 | 2.884 ± 0.444 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.034 ± 0.000 | 0.037 ± 0.000 | 0.041 ± 0.000 | 0.043 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.041 ± 0.000 | 0.045 ± 0.001 | 0.058 ± 0.001 | 0.064 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.053 ± 0.001 | 0.056 ± 0.001 | 0.067 ± 0.000 | 0.149 ± 0.059 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 236.986 ± 2.403 | 224.274 ± 1.977 | 261.231 ± 6.794 | 306.170 ± 3.753 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 2.135 ± 0.011 | 2.145 ± 0.008 | 2.293 ± 0.012 | 2.743 ± 0.151 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 10.099 ± 0.036 | 10.072 ± 0.091 | 11.274 ± 0.063 | 12.331 ± 0.279 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 46.421 ± 0.274 | 47.088 ± 0.485 | 50.423 ± 0.178 | 68.531 ± 0.241 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.538 ± 0.002 | 0.544 ± 0.004 | 0.567 ± 0.005 | 0.604 ± 0.042 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.025 ± 0.000 | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.036 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.082 ± 0.000 | 0.083 ± 0.000 | 0.088 ± 0.000 | 0.106 ± 0.006 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.118 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.115 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.120 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.118 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.120 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.066 ± 0.000 | 0.123 ± 0.002 | 0.034 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.046 ± 0.000 | 0.095 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.130 ± 0.000 | 0.202 ± 0.003 | 0.103 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.048 ± 0.000 | 0.097 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.052 ± 0.000 | 0.104 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.067 ± 0.000 | 0.106 ± 0.002 | 0.034 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.047 ± 0.000 | 0.083 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.131 ± 0.000 | 0.176 ± 0.003 | 0.105 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.048 ± 0.000 | 0.084 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.052 ± 0.000 | 0.088 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.055 ± 0.000 | 0.179 ± 0.005 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.049 ± 0.000 | 0.163 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.088 ± 0.001 | 0.220 ± 0.004 | 0.030 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.049 ± 0.000 | 0.162 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.050 ± 0.000 | 0.163 ± 0.003 | 0.018 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.055 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.049 ± 0.000 | 0.127 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.094 ± 0.000 | 0.169 ± 0.005 | 0.030 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.049 ± 0.000 | 0.129 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.050 ± 0.000 | 0.130 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.037 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.033 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.049 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.034 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.034 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.090 ± 0.000 | 0.128 ± 0.002 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.081 ± 0.000 | 0.113 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.121 ± 0.001 | 0.160 ± 0.005 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.082 ± 0.000 | 0.112 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.084 ± 0.000 | 0.115 ± 0.002 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.093 ± 0.000 | 0.132 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.082 ± 0.000 | 0.118 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.123 ± 0.000 | 0.162 ± 0.004 | 0.041 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.084 ± 0.000 | 0.118 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.085 ± 0.000 | 0.119 ± 0.002 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.056 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.051 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.064 ± 0.000 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.053 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.053 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.263 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.029 ± 0.000 | 0.255 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.038 ± 0.000 | 0.278 ± 0.003 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.028 ± 0.000 | 0.256 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.029 ± 0.000 | 0.256 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.115 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.028 ± 0.000 | 0.111 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.037 ± 0.000 | 0.132 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.028 ± 0.000 | 0.110 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.028 ± 0.000 | 0.112 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.068 ± 0.000 | 0.085 ± 0.000 | 0.060 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.038 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.182 ± 0.000 | 0.212 ± 0.002 | 0.178 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.042 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.053 ± 0.000 | 0.021 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.069 ± 0.000 | 0.206 ± 0.004 | 0.057 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.027 ± 0.000 | 0.144 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.183 ± 0.001 | 0.354 ± 0.003 | 0.163 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.030 ± 0.000 | 0.149 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.000 | 0.165 ± 0.003 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.069 ± 0.000 | 0.141 ± 0.001 | 0.057 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.026 ± 0.000 | 0.090 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.182 ± 0.000 | 0.272 ± 0.003 | 0.176 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.030 ± 0.000 | 0.094 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.000 | 0.104 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.027 ± 0.001 | 0.023 ± 0.000 | 0.022 ± 0.002 | 0.023 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.025 ± 0.000 | 0.029 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.048 ± 0.000 | 0.050 ± 0.000 | 0.052 ± 0.000 | 0.058 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.157 ± 0.001 | 0.160 ± 0.000 | 0.172 ± 0.001 | 0.166 ± 0.010 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.020 ± 0.000 | 0.023 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 5.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 34.4x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 35.0x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

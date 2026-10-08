# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261008_044146`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261008_044146`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_t1_20261008_044146.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261008_044146/cpu_ops_t1_20261008_044146.md`

## CPU Benchmark Items

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

## Cross-Backend Spread Audit

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

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Collection recovery

Completed einsum measurements retained from the initial collection; CPU ops were recollected after the input-restoration fix. Component commits and original metadata are recorded in `run.yaml` and `einsum_run.yaml`. Exact commands: `data/results/amd-cpu/cpu/refresh/20261008_044136/commands_remaining.sh`.

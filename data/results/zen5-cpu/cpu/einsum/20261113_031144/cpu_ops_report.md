# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_031144/cpu_ops_t4_20261113_031144.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 8.381 ± 1.032 | 8.396 ± 0.011 | 27.226 ± 0.039 | 17.735 ± 0.254 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.134 ± 0.000 | 0.132 ± 0.001 | 0.413 ± 0.002 | 0.379 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.072 ± 0.005 | 2.079 ± 0.017 | 6.857 ± 0.016 | 3.888 ± 1.011 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.520 ± 0.001 | 0.516 ± 0.000 | 1.697 ± 0.009 | 1.058 ± 0.152 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.521 ± 0.001 | 0.520 ± 0.002 | 0.503 ± 0.002 | 0.423 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.136 ± 0.000 | 0.134 ± 0.001 | 0.132 ± 0.000 | 0.109 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.038 ± 0.000 | 0.039 ± 0.000 | 0.029 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.737 ± 0.002 | 0.735 ± 0.001 | 2.628 ± 0.013 | 1.687 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.052 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.735 ± 0.001 | 0.735 ± 0.002 | 0.662 ± 0.002 | 0.655 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.188 ± 0.000 | 0.187 ± 0.001 | 0.172 ± 0.000 | 0.169 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 2.231 ± 0.003 | 2.225 ± 0.003 | 7.889 ± 0.025 | 3.135 ± 0.987 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.160 ± 0.037 | 0.141 ± 0.000 | 0.131 ± 0.001 | 0.146 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.564 ± 0.002 | 0.562 ± 0.001 | 1.992 ± 0.009 | 1.138 ± 0.188 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.146 ± 0.000 | 0.144 ± 0.000 | 0.500 ± 0.002 | 0.561 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.116 ± 0.001 | 0.114 ± 0.000 | 0.386 ± 0.022 | 1.512 ± 0.045 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.022 ± 0.001 | 0.155 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.033 ± 0.001 | 0.033 ± 0.001 | 0.096 ± 0.005 | 0.294 ± 0.018 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.039 ± 0.003 | 0.162 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.146 ± 0.001 | 0.148 ± 0.002 | 0.044 ± 0.001 | 0.154 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.147 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.040 ± 0.000 | 0.041 ± 0.000 | 0.022 ± 0.000 | 0.152 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.152 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.045 ± 0.000 | 0.044 ± 0.000 | 0.067 ± 0.001 | 0.283 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.003 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.154 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.045 ± 0.001 | 0.043 ± 0.000 | 0.052 ± 0.000 | 0.157 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.154 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.050 ± 0.001 | 0.050 ± 0.000 | 0.089 ± 0.005 | 0.457 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.154 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.035 ± 0.001 | 0.164 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.000 | 0.154 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.488 ± 0.007 | 2.381 ± 0.004 | 5.534 ± 0.051 | 4.366 ± 0.037 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.046 ± 0.001 | 0.044 ± 0.000 | 0.097 ± 0.001 | 0.134 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.604 ± 0.002 | 0.604 ± 0.004 | 1.408 ± 0.009 | 1.069 ± 0.070 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.157 ± 0.001 | 0.156 ± 0.001 | 0.366 ± 0.002 | 0.371 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.314 ± 0.001 | 0.314 ± 0.001 | 0.207 ± 0.001 | 0.326 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.084 ± 0.000 | 0.082 ± 0.000 | 0.057 ± 0.000 | 0.087 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.020 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.212 ± 0.002 | 0.208 ± 0.001 | 0.628 ± 0.002 | 0.944 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.017 ± 0.008 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.206 ± 0.000 | 0.213 ± 0.000 | 0.162 ± 0.000 | 0.241 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.055 ± 0.001 | 0.046 ± 0.000 | 0.065 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.648 ± 0.001 | 0.643 ± 0.001 | 1.625 ± 0.016 | 1.259 ± 0.062 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.055 ± 0.018 | 0.045 ± 0.000 | 0.033 ± 0.000 | 0.049 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.170 ± 0.000 | 0.168 ± 0.000 | 0.424 ± 0.002 | 0.683 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.165 ± 0.000 | 0.164 ± 0.001 | 0.111 ± 0.000 | 0.177 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.871 ± 0.009 | 0.666 ± 0.006 | 0.889 ± 0.020 | 2.167 ± 0.034 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.045 ± 0.000 | 0.044 ± 0.000 | 0.037 ± 0.001 | 0.059 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.157 ± 0.001 | 0.174 ± 0.002 | 0.248 ± 0.004 | 0.552 ± 0.041 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.046 ± 0.000 | 0.055 ± 0.000 | 0.085 ± 0.004 | 0.196 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.209 ± 0.001 | 0.161 ± 0.001 | 0.087 ± 0.001 | 0.337 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.059 ± 0.000 | 0.048 ± 0.000 | 0.037 ± 0.000 | 0.091 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.090 ± 0.000 | 0.114 ± 0.001 | 0.122 ± 0.004 | 0.488 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.087 ± 0.000 | 0.075 ± 0.000 | 0.046 ± 0.000 | 0.129 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.246 ± 0.001 | 0.263 ± 0.001 | 0.342 ± 0.011 | 0.924 ± 0.032 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.026 ± 0.011 | 0.023 ± 0.000 | 0.026 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.065 ± 0.000 | 0.074 ± 0.000 | 0.109 ± 0.001 | 0.256 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.063 ± 0.000 | 0.063 ± 0.001 | 0.042 ± 0.001 | 0.074 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 14.608 ± 0.014 | 14.462 ± 0.015 | 59.770 ± 0.129 | 30.167 ± 0.146 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.231 ± 0.001 | 0.230 ± 0.000 | 0.921 ± 0.003 | 0.530 ± 0.122 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 3.614 ± 0.011 | 3.614 ± 0.004 | 14.951 ± 0.076 | 5.793 ± 1.568 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.909 ± 0.001 | 0.901 ± 0.002 | 3.736 ± 0.010 | 1.831 ± 0.203 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.271 ± 0.002 | 1.266 ± 0.003 | 1.355 ± 0.002 | 1.539 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.031 ± 0.000 | 0.037 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.323 ± 0.000 | 0.321 ± 0.000 | 0.346 ± 0.001 | 0.392 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.087 ± 0.000 | 0.085 ± 0.000 | 0.094 ± 0.000 | 0.109 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.165 ± 0.009 | 1.159 ± 0.007 | 5.434 ± 0.024 | 2.036 ± 0.307 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.087 ± 0.024 | 0.076 ± 0.000 | 0.094 ± 0.000 | 0.091 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 1.154 ± 0.003 | 1.154 ± 0.002 | 1.364 ± 0.004 | 1.276 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.295 ± 0.001 | 0.294 ± 0.002 | 0.347 ± 0.001 | 0.322 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.772 ± 0.006 | 3.772 ± 0.005 | 18.299 ± 0.053 | 4.933 ± 3.384 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.240 ± 0.001 | 0.238 ± 0.001 | 0.293 ± 0.001 | 0.259 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.954 ± 0.005 | 0.952 ± 0.006 | 4.587 ± 0.011 | 1.958 ± 0.473 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.244 ± 0.002 | 0.243 ± 0.000 | 1.149 ± 0.002 | 0.705 ± 0.038 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.438 ± 0.004 | 10.055 ± 0.057 | 3.244 ± 0.076 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.262 ± 0.003 | 0.054 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.123 ± 0.001 | 2.607 ± 0.034 | 0.772 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.713 ± 0.005 | 0.201 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.480 ± 0.001 | 0.196 ± 0.001 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.108 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.131 ± 0.000 | 0.127 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.111 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.150 ± 0.000 | 0.271 ± 0.005 | 0.802 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.113 ± 0.001 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.137 ± 0.000 | 0.221 ± 0.002 | 0.205 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.134 ± 0.001 | 0.057 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.171 ± 0.000 | 7.086 ± 0.051 | 1.323 ± 0.020 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.222 ± 0.005 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.061 ± 0.000 | 1.890 ± 0.018 | 0.324 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.555 ± 0.008 | 0.088 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.610 ± 0.009 | 1.844 ± 0.038 | 3.949 ± 0.625 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.180 ± 0.001 | 0.125 ± 0.002 | 0.097 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.709 ± 0.006 | 0.574 ± 0.006 | 1.063 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.306 ± 0.001 | 0.268 ± 0.005 | 0.334 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.771 ± 0.003 | 0.210 ± 0.002 | 0.686 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.060 ± 0.000 | 0.077 ± 0.001 | 0.028 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.232 ± 0.003 | 0.111 ± 0.002 | 0.183 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.094 ± 0.000 | 0.087 ± 0.001 | 0.053 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.678 ± 0.003 | 0.340 ± 0.006 | 0.993 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.065 ± 0.000 | 0.081 ± 0.000 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.323 ± 0.000 | 0.141 ± 0.001 | 0.258 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.118 ± 0.000 | 0.094 ± 0.002 | 0.075 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.044 ± 0.006 | 0.864 ± 0.027 | 1.662 ± 0.103 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.097 ± 0.000 | 0.094 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.439 ± 0.002 | 0.345 ± 0.006 | 0.468 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.240 ± 0.000 | 0.143 ± 0.000 | 0.132 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 111.910 ± 0.280 | 111.791 ± 0.407 | 191.634 ± 47.546 | 270.963 ± 71.786 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 2.211 ± 0.007 | 2.214 ± 0.024 | 7.447 ± 2.001 | 41.990 ± 36.646 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 9.503 ± 0.034 | 9.445 ± 0.014 | 136.441 ± 130.174 | 18.366 ± 7.974 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 23.378 ± 0.032 | 23.239 ± 0.048 | 37.933 ± 5.712 | 89.913 ± 92.252 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.449 ± 0.001 | 0.447 ± 0.001 | 0.287 ± 0.002 | 0.459 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | 264.629 ± 0.985 | 212.834 ± 0.510 | 201.070 ± 1.065 | 252.555 ± 4.349 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 20.389 ± 0.063 | 15.247 ± 0.036 | 6.026 ± 0.030 | 11.294 ± 0.669 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 53.204 ± 0.194 | 45.632 ± 1.082 | 33.649 ± 0.170 | 52.137 ± 4.026 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | 150.893 ± 0.325 | 212.407 ± 0.761 | 140.443 ± 2.735 | 256.938 ± 9.147 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 10.416 ± 0.032 | 15.265 ± 0.010 | 5.231 ± 0.021 | 11.113 ± 0.458 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 29.361 ± 0.223 | 45.188 ± 0.428 | 25.244 ± 0.094 | 50.360 ± 2.551 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | 214.406 ± 0.982 | 104.323 ± 0.686 | 247.948 ± 1.427 | 180.203 ± 9.748 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 5.794 ± 0.072 | 2.886 ± 0.016 | 5.552 ± 0.100 | 5.586 ± 1.264 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 29.778 ± 0.105 | 16.220 ± 0.360 | 42.452 ± 0.494 | 29.706 ± 3.822 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | 199.116 ± 0.377 | 107.345 ± 0.293 | 116.522 ± 0.672 | 175.402 ± 7.990 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 4.401 ± 0.056 | 2.981 ± 0.013 | 2.685 ± 0.029 | 5.260 ± 0.676 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 27.230 ± 0.113 | 16.692 ± 0.173 | 18.293 ± 0.127 | 29.629 ± 8.474 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 19.179 ± 0.032 | 17.522 ± 0.028 | 27.701 ± 0.139 | 20.431 ± 0.859 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.079 ± 0.003 | 0.056 ± 0.000 | 0.085 ± 0.003 | 0.144 ± 0.017 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.392 ± 0.002 | 0.304 ± 0.001 | 0.436 ± 0.008 | 0.607 ± 0.063 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 2.516 ± 0.039 | 2.205 ± 0.005 | 4.482 ± 0.055 | 3.694 ± 0.860 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.031 ± 0.001 | 0.019 ± 0.000 | 0.028 ± 0.003 | 0.035 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 52.604 ± 0.085 | 77.467 ± 2.960 | 60.035 ± 3.427 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.158 ± 0.000 | 0.284 ± 0.004 | 0.278 ± 0.021 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.920 ± 0.001 | 1.310 ± 0.021 | 1.295 ± 0.104 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 6.700 ± 0.014 | 11.533 ± 0.067 | 8.521 ± 0.558 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.000 | 0.100 ± 0.002 | 0.082 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | 279.670 ± 0.612 | 159.958 ± 0.396 | 161.664 ± 0.341 | 262.896 ± 7.583 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 7.913 ± 0.029 | 4.235 ± 0.026 | 4.259 ± 0.063 | 9.353 ± 1.703 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 40.770 ± 0.173 | 25.083 ± 0.506 | 29.209 ± 0.155 | 44.872 ± 4.594 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | 233.671 ± 0.668 | 159.224 ± 4.442 | 158.557 ± 1.405 | 248.021 ± 9.939 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 5.057 ± 0.016 | 4.230 ± 0.032 | 4.536 ± 0.082 | 9.047 ± 0.865 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 31.903 ± 0.061 | 24.122 ± 0.157 | 32.748 ± 0.224 | 46.782 ± 5.473 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 29.586 ± 0.088 | 38.444 ± 8.063 | 76.962 ± 108.321 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.379 ± 0.001 | 0.264 ± 0.003 | 0.214 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.249 ± 0.011 | 31.899 ± 8.991 | 1.928 ± 3.737 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 5.175 ± 0.042 | 5.740 ± 0.010 | 13.126 ± 43.707 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.138 ± 0.001 | 0.115 ± 0.002 | 0.068 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | 21.911 ± 0.171 | 20.287 ± 0.035 | 13.409 ± 0.095 | 20.361 ± 0.194 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.158 ± 0.042 | 0.809 ± 0.006 | 0.786 ± 0.030 | 0.756 ± 0.022 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 4.341 ± 0.133 | 3.455 ± 0.057 | 2.758 ± 0.117 | 4.223 ± 0.038 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | 11.596 ± 0.106 | 19.585 ± 0.066 | 22.972 ± 0.254 | 19.702 ± 0.218 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.605 ± 0.016 | 0.818 ± 0.006 | 0.905 ± 0.036 | 0.757 ± 0.047 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.346 ± 0.088 | 3.476 ± 0.049 | 4.162 ± 0.014 | 4.265 ± 0.066 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 429.050 ± 1.373 | 396.728 ± 5.966 | 526.895 ± 9.607 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.850 ± 0.011 | 3.277 ± 0.032 | 4.081 ± 0.149 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 27.138 ± 0.058 | 15.400 ± 0.179 | 19.023 ± 0.765 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 77.782 ± 0.198 | 83.227 ± 1.276 | 109.312 ± 1.315 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.216 ± 0.003 | 0.815 ± 0.012 | 0.807 ± 0.015 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | 567.500 ± 11.754 | 406.784 ± 2.792 | 430.255 ± 1.087 | 522.954 ± 8.597 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 37.516 ± 0.071 | 27.124 ± 0.078 | 16.492 ± 0.034 | 19.029 ± 0.308 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 95.203 ± 0.301 | 77.584 ± 0.267 | 84.846 ± 0.230 | 108.444 ± 3.493 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | 295.189 ± 0.525 | 429.216 ± 0.861 | 350.452 ± 1.016 | 531.103 ± 9.847 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 18.817 ± 0.075 | 27.136 ± 0.040 | 15.334 ± 0.047 | 19.001 ± 0.235 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 50.124 ± 0.209 | 77.901 ± 0.283 | 78.074 ± 1.864 | 108.108 ± 3.027 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 17.458 ± 0.025 | 17.456 ± 0.009 | 22.427 ± 0.157 | 19.526 ± 0.878 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.051 ± 0.000 | 0.052 ± 0.000 | 0.097 ± 0.003 | 0.117 ± 0.016 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.299 ± 0.001 | 0.298 ± 0.001 | 0.459 ± 0.007 | 0.607 ± 0.110 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 2.193 ± 0.007 | 2.192 ± 0.005 | 3.932 ± 0.040 | 2.928 ± 0.573 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 4.332 ± 0.013 | 4.326 ± 0.008 | 5.860 ± 0.032 | 5.443 ± 0.368 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.169 ± 0.005 | 1.165 ± 0.004 | 1.552 ± 0.026 | 1.606 ± 0.326 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 42.902 ± 0.113 | 43.114 ± 0.887 | 68.053 ± 0.484 | 87.111 ± 2.538 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.557 ± 0.000 | 0.555 ± 0.002 | 0.494 ± 0.005 | 0.854 ± 0.008 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.397 ± 0.004 | 1.368 ± 0.004 | 2.141 ± 0.008 | 3.210 ± 0.060 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 7.148 ± 0.008 | 7.169 ± 0.107 | 12.846 ± 0.083 | 17.948 ± 0.140 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.136 ± 0.001 | 0.134 ± 0.000 | 0.142 ± 0.001 | 0.118 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 9.103 ± 0.037 | 8.994 ± 0.029 | 41.739 ± 18.921 | 27.080 ± 1.475 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 9.787 ± 0.044 | 9.572 ± 0.042 | 54.543 ± 17.877 | 42.185 ± 115.044 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 11.712 ± 0.023 | 11.335 ± 0.039 | 64.219 ± 53.576 | 119.273 ± 20.181 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.108 ± 0.000 | 0.107 ± 0.001 | 0.177 ± 0.002 | 0.178 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.134 ± 0.003 | 0.125 ± 0.003 | 2.256 ± 0.005 | 0.279 ± 1.847 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.223 ± 0.006 | 0.177 ± 0.002 | 7.261 ± 1.045 | 0.327 ± 0.018 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.404 ± 0.001 | 0.376 ± 0.009 | 20.944 ± 1.012 | 4.121 ± 63.690 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.486 ± 0.005 | 0.433 ± 0.011 | 32.683 ± 2.004 | 4.060 ± 128.776 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.731 ± 0.009 | 0.597 ± 0.003 | 29.804 ± 2.962 | 2.652 ± 4.145 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 1.782 ± 0.033 | 1.618 ± 0.017 | 2.056 ± 0.066 | 11.104 ± 125.108 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 2.008 ± 0.015 | 1.801 ± 0.022 | 5.379 ± 2.980 | 126.612 ± 127.163 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 2.777 ± 0.048 | 2.453 ± 0.013 | 3.738 ± 0.089 | 18.680 ± 114.649 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.034 ± 0.000 | 0.035 ± 0.000 | 0.053 ± 0.001 | 0.050 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.044 ± 0.000 | 0.039 ± 0.001 | 0.064 ± 0.001 | 0.104 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.081 ± 0.001 | 0.058 ± 0.001 | 0.088 ± 0.001 | 0.138 ± 0.008 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 257.153 ± 0.690 | 241.401 ± 1.014 | 329.146 ± 5.954 | 419.406 ± 4.075 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 3.292 ± 0.007 | 3.272 ± 0.005 | 2.896 ± 0.015 | 3.327 ± 0.019 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 17.939 ± 0.073 | 17.913 ± 0.020 | 14.188 ± 0.111 | 15.555 ± 0.038 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 44.770 ± 0.103 | 44.736 ± 0.078 | 63.626 ± 0.267 | 92.011 ± 0.466 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.829 ± 0.001 | 0.824 ± 0.002 | 0.709 ± 0.005 | 0.755 ± 0.010 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.034 ± 0.000 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.123 ± 0.000 | 0.122 ± 0.000 | 0.111 ± 0.000 | 0.125 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.153 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.144 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.149 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.148 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.152 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | 0.309 ± 0.004 | 0.086 ± 0.000 | 0.158 ± 0.005 | 0.042 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.234 ± 0.003 | 0.033 ± 0.000 | 0.122 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | 0.501 ± 0.003 | 0.243 ± 0.001 | 0.259 ± 0.005 | 0.133 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.245 ± 0.002 | 0.037 ± 0.000 | 0.124 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.263 ± 0.002 | 0.048 ± 0.000 | 0.132 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | 0.096 ± 0.001 | 0.085 ± 0.000 | 0.135 ± 0.001 | 0.043 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.063 ± 0.000 | 0.032 ± 0.000 | 0.105 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | 0.192 ± 0.001 | 0.238 ± 0.000 | 0.225 ± 0.006 | 0.134 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.066 ± 0.000 | 0.036 ± 0.000 | 0.105 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.072 ± 0.000 | 0.047 ± 0.000 | 0.111 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | 0.931 ± 0.011 | 0.051 ± 0.000 | 0.230 ± 0.008 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.900 ± 0.017 | 0.042 ± 0.000 | 0.206 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | 0.984 ± 0.010 | 0.078 ± 0.000 | 0.281 ± 0.007 | 0.053 ± 0.010 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.903 ± 0.006 | 0.043 ± 0.000 | 0.204 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.919 ± 0.011 | 0.045 ± 0.000 | 0.209 ± 0.004 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | 0.185 ± 0.002 | 0.048 ± 0.000 | 0.175 ± 0.005 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.169 ± 0.002 | 0.038 ± 0.000 | 0.161 ± 0.007 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | 0.228 ± 0.002 | 0.076 ± 0.000 | 0.216 ± 0.005 | 0.044 ± 0.008 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.171 ± 0.001 | 0.039 ± 0.000 | 0.162 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.174 ± 0.002 | 0.041 ± 0.000 | 0.165 ± 0.005 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.048 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.042 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.062 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | 1.335 ± 0.005 | 0.077 ± 0.000 | 0.162 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 1.255 ± 0.012 | 0.054 ± 0.000 | 0.142 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | 1.398 ± 0.015 | 0.140 ± 0.001 | 0.205 ± 0.004 | 0.069 ± 0.024 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 1.291 ± 0.023 | 0.055 ± 0.000 | 0.142 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 1.299 ± 0.010 | 0.060 ± 0.000 | 0.149 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | 0.164 ± 0.002 | 0.075 ± 0.000 | 0.168 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.146 ± 0.002 | 0.053 ± 0.000 | 0.149 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | 0.225 ± 0.002 | 0.139 ± 0.000 | 0.207 ± 0.006 | 0.069 ± 0.030 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.146 ± 0.001 | 0.055 ± 0.000 | 0.148 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.150 ± 0.002 | 0.059 ± 0.000 | 0.151 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.058 ± 0.000 | 0.071 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.065 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.000 | 0.083 ± 0.001 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.068 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.068 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | 0.172 ± 0.005 | 0.039 ± 0.000 | 0.335 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.157 ± 0.004 | 0.033 ± 0.000 | 0.326 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | 0.181 ± 0.004 | 0.050 ± 0.000 | 0.354 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.162 ± 0.005 | 0.033 ± 0.000 | 0.325 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.165 ± 0.002 | 0.035 ± 0.000 | 0.328 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | 0.072 ± 0.001 | 0.039 ± 0.000 | 0.145 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.067 ± 0.001 | 0.033 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | 0.080 ± 0.001 | 0.048 ± 0.000 | 0.165 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.067 ± 0.001 | 0.033 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.069 ± 0.001 | 0.035 ± 0.000 | 0.139 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.000 | 0.108 ± 0.001 | 0.072 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.047 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.312 ± 0.000 | 0.266 ± 0.002 | 0.206 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.053 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.067 ± 0.000 | 0.023 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | 0.321 ± 0.007 | 0.100 ± 0.000 | 0.263 ± 0.006 | 0.068 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.192 ± 0.002 | 0.023 ± 0.000 | 0.185 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | 0.640 ± 0.003 | 0.313 ± 0.001 | 0.450 ± 0.005 | 0.203 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.205 ± 0.003 | 0.029 ± 0.000 | 0.190 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.229 ± 0.003 | 0.044 ± 0.000 | 0.210 ± 0.004 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | 0.117 ± 0.000 | 0.100 ± 0.000 | 0.178 ± 0.005 | 0.069 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.060 ± 0.001 | 0.022 ± 0.000 | 0.113 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | 0.277 ± 0.003 | 0.312 ± 0.000 | 0.348 ± 0.009 | 0.203 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.063 ± 0.001 | 0.028 ± 0.000 | 0.119 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.074 ± 0.002 | 0.043 ± 0.000 | 0.134 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.001 | 0.035 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.032 ± 0.000 | 0.033 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.029 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.063 ± 0.000 | 0.061 ± 0.000 | 0.065 ± 0.000 | 0.073 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.213 ± 0.000 | 0.212 ± 0.001 | 0.216 ± 0.001 | 0.204 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 13.1x faster than the slowest successful cell (`jax-cpu`, 1.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 1.512 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.4x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 10.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 2.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.713 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 0.480 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 41.5x faster than the slowest successful cell (`pytorch-cpu`, 7.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 30.9x faster than the slowest successful cell (`pytorch-cpu`, 1.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.555 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 19.0x faster than the slowest successful cell (`jax-cpu`, 41.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 41.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 136.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 136.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 31.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 31.899 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=64`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`jax-cpu`, 119.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=64`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 119.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-eager` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 2.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 2.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `jax-cpu` is 22.2x faster than the slowest successful cell (`pytorch-cpu`, 7.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-eager` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 7.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-trace` is 41.1x faster than the slowest successful cell (`pytorch-cpu`, 7.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-eager` is 51.8x faster than the slowest successful cell (`pytorch-cpu`, 20.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 55.7x faster than the slowest successful cell (`pytorch-cpu`, 20.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-eager` is 67.3x faster than the slowest successful cell (`pytorch-cpu`, 32.683 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-trace` is 75.5x faster than the slowest successful cell (`pytorch-cpu`, 32.683 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 29.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-eager` is 40.8x faster than the slowest successful cell (`pytorch-cpu`, 29.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-trace` is 50.0x faster than the slowest successful cell (`pytorch-cpu`, 29.804 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `pytorch-cpu` is 23.5x faster than the slowest successful cell (`jax-cpu`, 126.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-eager` is 63.1x faster than the slowest successful cell (`jax-cpu`, 126.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-trace` is 70.3x faster than the slowest successful cell (`jax-cpu`, 126.612 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 29.6x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 30.6x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.8x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.234 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.5x faster than the slowest successful cell (`tenferro-eager`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 45.7x faster than the slowest successful cell (`tenferro-eager`, 0.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`tenferro-eager`, 0.931 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 64.6x faster than the slowest successful cell (`tenferro-eager`, 0.900 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 0.900 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`32x32`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 0.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 59.5x faster than the slowest successful cell (`tenferro-eager`, 0.903 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 0.903 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 42.0x faster than the slowest successful cell (`tenferro-eager`, 0.919 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.919 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 70.0x faster than the slowest successful cell (`tenferro-eager`, 1.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 1.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 93.3x faster than the slowest successful cell (`tenferro-eager`, 1.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 1.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`32x32`): `jax-cpu` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 1.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 84.4x faster than the slowest successful cell (`tenferro-eager`, 1.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 1.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 64.0x faster than the slowest successful cell (`tenferro-eager`, 1.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 1.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.326 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

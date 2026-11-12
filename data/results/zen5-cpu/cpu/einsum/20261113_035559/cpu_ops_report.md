# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

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

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_035559/cpu_ops_t8_20261113_035559.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 4.177 ± 0.009 | 4.162 ± 0.006 | 27.309 ± 0.079 | 8.039 ± 0.496 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.073 ± 0.000 | 0.071 ± 0.000 | 0.413 ± 0.001 | 0.390 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 1.053 ± 0.004 | 1.048 ± 0.003 | 6.943 ± 0.023 | 2.748 ± 0.340 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.266 ± 0.000 | 0.264 ± 0.001 | 1.694 ± 0.007 | 0.592 ± 0.030 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.524 ± 0.003 | 0.526 ± 0.001 | 0.506 ± 0.001 | 0.474 ± 0.053 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.136 ± 0.000 | 0.136 ± 0.000 | 0.132 ± 0.002 | 0.111 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.038 ± 0.000 | 0.040 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.378 ± 0.002 | 0.378 ± 0.001 | 2.645 ± 0.026 | 1.722 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.052 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.739 ± 0.004 | 0.737 ± 0.002 | 0.664 ± 0.003 | 0.672 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.184 ± 0.001 | 0.188 ± 0.000 | 0.173 ± 0.001 | 0.170 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 1.132 ± 0.002 | 1.136 ± 0.001 | 7.943 ± 0.025 | 3.680 ± 0.037 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.141 ± 0.000 | 0.141 ± 0.000 | 0.132 ± 0.001 | 0.147 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.289 ± 0.001 | 0.288 ± 0.000 | 1.975 ± 0.005 | 0.867 ± 0.019 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.078 ± 0.000 | 0.077 ± 0.000 | 0.499 ± 0.001 | 0.567 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 0.059 ± 0.000 | 0.060 ± 0.000 | 0.255 ± 0.035 | 1.517 ± 0.025 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.023 ± 0.001 | 0.158 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.077 ± 0.003 | 0.333 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.037 ± 0.002 | 0.167 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.148 ± 0.001 | 0.149 ± 0.000 | 0.045 ± 0.000 | 0.157 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.149 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.041 ± 0.000 | 0.041 ± 0.000 | 0.023 ± 0.001 | 0.153 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.152 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.071 ± 0.000 | 0.283 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.155 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.048 ± 0.001 | 0.044 ± 0.000 | 0.053 ± 0.001 | 0.158 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.001 | 0.153 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.029 ± 0.001 | 0.029 ± 0.000 | 0.082 ± 0.004 | 0.472 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.156 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.034 ± 0.001 | 0.169 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.001 | 0.158 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 1.230 ± 0.004 | 1.215 ± 0.005 | 5.591 ± 0.039 | 2.586 ± 0.169 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.027 ± 0.000 | 0.099 ± 0.002 | 0.136 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.319 ± 0.001 | 0.309 ± 0.003 | 1.430 ± 0.008 | 0.571 ± 0.024 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.085 ± 0.001 | 0.083 ± 0.001 | 0.374 ± 0.004 | 0.376 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.316 ± 0.002 | 0.316 ± 0.001 | 0.205 ± 0.002 | 0.328 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.084 ± 0.000 | 0.083 ± 0.000 | 0.058 ± 0.001 | 0.087 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.112 ± 0.001 | 0.109 ± 0.000 | 0.629 ± 0.003 | 0.955 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.206 ± 0.001 | 0.206 ± 0.001 | 0.162 ± 0.001 | 0.243 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.056 ± 0.000 | 0.047 ± 0.001 | 0.066 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.340 ± 0.013 | 0.333 ± 0.002 | 1.635 ± 0.010 | 1.009 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.047 ± 0.000 | 0.045 ± 0.000 | 0.035 ± 0.001 | 0.049 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.091 ± 0.000 | 0.092 ± 0.000 | 0.423 ± 0.002 | 0.691 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.165 ± 0.000 | 0.164 ± 0.001 | 0.113 ± 0.001 | 0.177 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.302 ± 0.004 | 0.395 ± 0.004 | 0.788 ± 0.021 | 2.061 ± 0.031 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.044 ± 0.001 | 0.045 ± 0.000 | 0.038 ± 0.002 | 0.061 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.083 ± 0.000 | 0.112 ± 0.001 | 0.228 ± 0.006 | 0.533 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.029 ± 0.000 | 0.039 ± 0.000 | 0.083 ± 0.004 | 0.205 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.209 ± 0.000 | 0.164 ± 0.002 | 0.085 ± 0.001 | 0.359 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.059 ± 0.000 | 0.049 ± 0.000 | 0.039 ± 0.001 | 0.096 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.053 ± 0.000 | 0.072 ± 0.000 | 0.119 ± 0.001 | 0.511 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.087 ± 0.001 | 0.077 ± 0.000 | 0.046 ± 0.000 | 0.134 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.124 ± 0.001 | 0.154 ± 0.003 | 0.320 ± 0.006 | 0.982 ± 0.027 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.028 ± 0.001 | 0.022 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.039 ± 0.000 | 0.048 ± 0.000 | 0.105 ± 0.002 | 0.265 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.000 | 0.060 ± 0.000 | 0.042 ± 0.001 | 0.076 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 7.316 ± 0.020 | 7.298 ± 0.035 | 60.123 ± 0.111 | 15.482 ± 0.115 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.123 ± 0.001 | 0.120 ± 0.000 | 0.923 ± 0.002 | 0.382 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 1.856 ± 0.003 | 1.826 ± 0.003 | 15.023 ± 0.024 | 3.954 ± 0.930 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.463 ± 0.001 | 0.465 ± 0.001 | 3.735 ± 0.010 | 1.031 ± 0.039 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 1.273 ± 0.003 | 1.270 ± 0.004 | 1.355 ± 0.006 | 1.532 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.031 ± 0.000 | 0.038 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.322 ± 0.001 | 0.322 ± 0.000 | 0.346 ± 0.004 | 0.388 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.087 ± 0.000 | 0.085 ± 0.000 | 0.094 ± 0.000 | 0.109 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.594 ± 0.002 | 0.591 ± 0.004 | 5.465 ± 0.028 | 1.660 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.078 ± 0.000 | 0.077 ± 0.000 | 0.094 ± 0.000 | 0.092 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 1.154 ± 0.001 | 1.172 ± 0.002 | 1.364 ± 0.005 | 1.277 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.294 ± 0.001 | 0.294 ± 0.000 | 0.349 ± 0.002 | 0.322 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 1.938 ± 0.002 | 1.917 ± 0.003 | 18.373 ± 0.062 | 3.921 ± 1.019 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.239 ± 0.001 | 0.239 ± 0.000 | 0.293 ± 0.001 | 0.260 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.489 ± 0.003 | 0.485 ± 0.001 | 4.586 ± 0.013 | 0.828 ± 0.057 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.129 ± 0.000 | 0.128 ± 0.001 | 1.151 ± 0.004 | 0.697 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.230 ± 0.006 | 9.364 ± 0.031 | 3.369 ± 0.081 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.263 ± 0.005 | 0.056 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.078 ± 0.001 | 2.620 ± 0.020 | 0.819 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.733 ± 0.020 | 0.206 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.487 ± 0.001 | 0.198 ± 0.002 | 0.026 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.108 ± 0.001 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.134 ± 0.000 | 0.127 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.111 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.090 ± 0.001 | 0.278 ± 0.004 | 0.811 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.112 ± 0.001 | 0.020 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.139 ± 0.000 | 0.222 ± 0.006 | 0.209 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.133 ± 0.001 | 0.059 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.107 ± 0.000 | 7.215 ± 0.027 | 1.321 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.222 ± 0.001 | 0.026 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 1.960 ± 0.022 | 0.331 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.550 ± 0.004 | 0.090 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.681 ± 0.007 | 1.735 ± 0.057 | 4.025 ± 0.195 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.180 ± 0.001 | 0.126 ± 0.002 | 0.099 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.475 ± 0.002 | 0.560 ± 0.017 | 1.044 ± 0.030 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.256 ± 0.002 | 0.275 ± 0.010 | 0.351 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.785 ± 0.006 | 0.214 ± 0.005 | 0.725 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.000 | 0.079 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.234 ± 0.000 | 0.114 ± 0.002 | 0.191 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.097 ± 0.000 | 0.088 ± 0.001 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.553 ± 0.001 | 0.349 ± 0.008 | 1.025 ± 0.017 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.067 ± 0.000 | 0.083 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.326 ± 0.001 | 0.145 ± 0.002 | 0.268 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.120 ± 0.000 | 0.095 ± 0.001 | 0.079 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.663 ± 0.005 | 0.879 ± 0.014 | 1.855 ± 0.024 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.099 ± 0.000 | 0.097 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.356 ± 0.001 | 0.350 ± 0.007 | 0.482 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.241 ± 0.000 | 0.146 ± 0.003 | 0.134 ± 0.001 | - | - |
| large | `eigh` | f64 | 8 | `1024x1024` | - | 88.866 ± 0.061 | 88.637 ± 0.087 | 218.733 ± 29.833 | 254.892 ± 89.319 | - | - |
| large | `eigh` | f64 | 8 | `128x128` | - | 2.213 ± 0.006 | 2.215 ± 0.003 | 34.814 ± 32.898 | 4.965 ± 30.923 | - | - |
| large | `eigh` | f64 | 8 | `256x256` | - | 9.615 ± 0.047 | 9.452 ± 0.012 | 150.846 ± 16.063 | 97.620 ± 146.663 | - | - |
| large | `eigh` | f64 | 8 | `512x512` | - | 21.040 ± 0.037 | 20.926 ± 0.040 | 159.687 ± 31.937 | 159.786 ± 17.892 | - | - |
| large | `eigh` | f64 | 8 | `64x64` | - | 0.450 ± 0.001 | 0.449 ± 0.002 | 0.284 ± 0.002 | 0.465 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | - | 200.217 ± 0.173 | 172.289 ± 0.332 | 133.519 ± 1.105 | 208.850 ± 10.311 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | - | 20.427 ± 0.115 | 15.361 ± 0.197 | 4.819 ± 0.037 | 11.752 ± 1.473 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | - | 46.356 ± 0.162 | 41.044 ± 0.039 | 21.747 ± 0.134 | 47.047 ± 5.961 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | - | 111.592 ± 0.339 | 172.219 ± 0.353 | 96.246 ± 1.156 | 202.972 ± 8.437 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | - | 10.387 ± 0.072 | 15.195 ± 0.046 | 4.305 ± 0.046 | 10.660 ± 2.244 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | - | 24.962 ± 0.036 | 40.871 ± 0.123 | 17.196 ± 0.045 | 47.552 ± 4.173 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | - | 132.679 ± 0.494 | 65.311 ± 0.170 | 144.877 ± 0.647 | 125.472 ± 15.277 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | - | 5.170 ± 0.053 | 2.384 ± 0.018 | 4.080 ± 0.052 | 5.679 ± 1.842 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | - | 20.136 ± 0.063 | 11.101 ± 0.043 | 27.513 ± 0.302 | 24.154 ± 5.345 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | - | 117.497 ± 0.721 | 67.514 ± 0.306 | 72.069 ± 0.326 | 134.454 ± 19.718 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | - | 3.521 ± 0.044 | 2.480 ± 0.006 | 2.088 ± 0.015 | 5.188 ± 1.414 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | - | 18.001 ± 0.105 | 11.795 ± 0.065 | 13.392 ± 0.261 | 23.335 ± 3.443 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `1024x1024` | - | 10.843 ± 0.029 | 9.281 ± 0.014 | 15.654 ± 0.138 | 11.865 ± 1.318 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `128x128` | - | 0.072 ± 0.001 | 0.044 ± 0.000 | 0.062 ± 0.001 | 0.105 ± 0.004 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `256x256` | - | 0.278 ± 0.003 | 0.179 ± 0.001 | 0.267 ± 0.002 | 0.376 ± 0.014 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `512x512` | - | 1.518 ± 0.008 | 1.180 ± 0.009 | 2.880 ± 0.034 | 2.029 ± 0.068 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `64x64` | - | 0.033 ± 0.001 | 0.018 ± 0.000 | 0.023 ± 0.001 | 0.038 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 27.496 ± 0.059 | 44.200 ± 3.261 | 32.450 ± 1.321 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.128 ± 0.000 | 0.198 ± 0.002 | 0.219 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.533 ± 0.002 | 0.799 ± 0.014 | 0.812 ± 0.026 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.536 ± 0.015 | 7.174 ± 0.056 | 4.984 ± 0.124 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.000 | 0.090 ± 0.003 | 0.084 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | - | 166.417 ± 0.551 | 96.686 ± 0.319 | 112.283 ± 0.481 | 196.751 ± 17.818 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | - | 7.304 ± 0.136 | 3.492 ± 0.019 | 3.407 ± 0.025 | 9.983 ± 2.996 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | - | 26.913 ± 0.146 | 15.900 ± 0.076 | 22.242 ± 0.235 | 41.499 ± 8.487 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | - | 137.590 ± 0.515 | 95.811 ± 0.308 | 106.932 ± 0.333 | 196.039 ± 20.053 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | - | 3.959 ± 0.034 | 3.485 ± 0.037 | 3.667 ± 0.059 | 10.002 ± 2.643 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | - | 20.256 ± 0.083 | 15.696 ± 0.067 | 24.083 ± 0.098 | 43.270 ± 10.476 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 21.021 ± 0.081 | 116.506 ± 57.985 | 134.703 ± 16.670 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.384 ± 0.001 | 0.273 ± 0.003 | 0.224 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.169 ± 0.004 | 27.912 ± 2.042 | 1.810 ± 2.913 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.408 ± 0.045 | 87.723 ± 56.017 | 133.156 ± 136.703 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.140 ± 0.001 | 0.118 ± 0.002 | 0.071 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | - | 16.085 ± 0.090 | 13.904 ± 0.157 | 9.124 ± 0.158 | 14.977 ± 1.396 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | - | 1.153 ± 0.020 | 0.748 ± 0.024 | 0.792 ± 0.029 | 0.590 ± 0.014 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | - | 4.055 ± 0.102 | 2.972 ± 0.051 | 2.566 ± 0.066 | 3.981 ± 0.209 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | - | 8.790 ± 0.044 | 13.657 ± 0.047 | 14.776 ± 0.086 | 14.341 ± 1.755 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | - | 0.633 ± 0.010 | 0.763 ± 0.021 | 0.891 ± 0.031 | 0.593 ± 0.014 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | - | 2.228 ± 0.122 | 2.875 ± 0.016 | 3.530 ± 0.033 | 3.630 ± 0.064 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 325.489 ± 1.387 | 313.254 ± 8.249 | 461.919 ± 8.161 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.997 ± 0.030 | 3.061 ± 0.031 | 4.256 ± 0.038 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 27.524 ± 0.059 | 14.258 ± 0.099 | 20.246 ± 1.455 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 67.896 ± 0.210 | 72.268 ± 1.763 | 101.664 ± 4.044 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.224 ± 0.003 | 0.802 ± 0.012 | 0.823 ± 0.012 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | - | 424.268 ± 10.693 | 303.631 ± 0.826 | 324.881 ± 1.221 | 469.782 ± 40.800 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | - | 38.089 ± 0.107 | 27.550 ± 0.091 | 15.216 ± 0.100 | 19.751 ± 1.986 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | - | 81.423 ± 0.194 | 67.739 ± 0.105 | 72.928 ± 0.293 | 104.819 ± 5.395 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | - | 215.030 ± 0.640 | 327.511 ± 1.021 | 273.352 ± 2.462 | 472.943 ± 11.997 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | - | 19.105 ± 0.119 | 27.513 ± 0.033 | 14.221 ± 0.051 | 20.098 ± 0.933 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | - | 42.196 ± 0.098 | 67.777 ± 0.067 | 72.558 ± 0.194 | 104.024 ± 5.156 | - | - |
| large | `matmul` | f64 | 8 | `1024x1024` | - | 9.048 ± 0.018 | 8.998 ± 0.015 | 12.798 ± 0.108 | 10.409 ± 1.251 | - | - |
| large | `matmul` | f64 | 8 | `128x128` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.071 ± 0.001 | 0.076 ± 0.007 | - | - |
| large | `matmul` | f64 | 8 | `256x256` | - | 0.177 ± 0.001 | 0.171 ± 0.002 | 0.297 ± 0.006 | 0.346 ± 0.058 | - | - |
| large | `matmul` | f64 | 8 | `512x512` | - | 1.162 ± 0.005 | 1.156 ± 0.005 | 2.254 ± 0.028 | 1.776 ± 0.116 | - | - |
| large | `matmul_rect` | f64 | 8 | `1024x256 * 256x1024` | - | 2.282 ± 0.009 | 2.279 ± 0.014 | 3.491 ± 0.020 | 3.337 ± 0.065 | - | - |
| large | `matmul_rect` | f64 | 8 | `256x1024 * 1024x256` | - | 0.677 ± 0.005 | 0.673 ± 0.005 | 0.926 ± 0.010 | 1.093 ± 0.045 | - | - |
| large | `qr` | f64 | 8 | `1024x1024` | - | 26.234 ± 0.149 | 26.228 ± 0.071 | 48.174 ± 1.086 | 75.888 ± 12.873 | - | - |
| large | `qr` | f64 | 8 | `128x128` | - | 0.560 ± 0.002 | 0.557 ± 0.003 | 0.452 ± 0.003 | 0.887 ± 0.006 | - | - |
| large | `qr` | f64 | 8 | `256x256` | - | 1.233 ± 0.007 | 1.204 ± 0.012 | 1.870 ± 0.010 | 3.429 ± 0.109 | - | - |
| large | `qr` | f64 | 8 | `512x512` | - | 4.911 ± 0.035 | 4.842 ± 0.030 | 9.519 ± 0.052 | 16.754 ± 0.573 | - | - |
| large | `qr` | f64 | 8 | `64x64` | - | 0.137 ± 0.001 | 0.136 ± 0.001 | 0.161 ± 0.001 | 0.122 ± 0.004 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=1` | - | 6.629 ± 0.055 | 6.233 ± 0.178 | 123.060 ± 38.034 | 132.370 ± 16.857 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=16` | - | 7.213 ± 0.054 | 6.717 ± 0.165 | 117.262 ± 35.254 | 146.012 ± 47.155 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=64` | - | 8.974 ± 0.044 | 8.263 ± 0.177 | 103.653 ± 63.889 | 127.997 ± 17.856 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=1` | - | 0.107 ± 0.000 | 0.108 ± 0.001 | 0.181 ± 0.002 | 0.189 ± 0.004 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=16` | - | 0.137 ± 0.001 | 0.125 ± 0.002 | 8.826 ± 0.014 | 5.521 ± 0.108 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=64` | - | 0.219 ± 0.005 | 0.175 ± 0.002 | 6.855 ± 0.999 | 0.320 ± 0.018 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=1` | - | 0.369 ± 0.004 | 0.329 ± 0.012 | 23.937 ± 3.999 | 2.468 ± 2.614 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=16` | - | 0.446 ± 0.005 | 0.391 ± 0.010 | 35.655 ± 5.994 | 1.436 ± 6.325 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=64` | - | 0.671 ± 0.005 | 0.546 ± 0.014 | 30.798 ± 6.031 | 14.999 ± 8.001 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=1` | - | 1.434 ± 0.021 | 1.282 ± 0.070 | 86.860 ± 67.015 | 131.334 ± 149.463 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=16` | - | 1.665 ± 0.013 | 1.495 ± 0.071 | 61.113 ± 69.968 | 140.936 ± 142.536 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=64` | - | 2.395 ± 0.013 | 2.023 ± 0.085 | 48.222 ± 21.181 | 119.716 ± 128.204 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=1` | - | 0.035 ± 0.000 | 0.037 ± 0.000 | 0.054 ± 0.001 | 0.052 ± 0.001 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=16` | - | 0.044 ± 0.000 | 0.041 ± 0.001 | 0.065 ± 0.003 | 0.096 ± 0.001 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=64` | - | 0.083 ± 0.001 | 0.059 ± 0.001 | 0.080 ± 0.001 | 0.124 ± 0.003 | - | - |
| large | `svd` | f64 | 8 | `1024x1024` | - | 192.505 ± 0.438 | 177.560 ± 0.739 | 266.351 ± 4.850 | 390.613 ± 7.649 | - | - |
| large | `svd` | f64 | 8 | `128x128` | - | 3.304 ± 0.023 | 3.364 ± 0.020 | 2.831 ± 0.004 | 3.377 ± 0.035 | - | - |
| large | `svd` | f64 | 8 | `256x256` | - | 18.247 ± 0.078 | 18.147 ± 0.080 | 13.350 ± 0.019 | 15.648 ± 0.155 | - | - |
| large | `svd` | f64 | 8 | `512x512` | - | 38.869 ± 0.096 | 38.713 ± 0.079 | 54.701 ± 0.086 | 88.783 ± 3.527 | - | - |
| large | `svd` | f64 | 8 | `64x64` | - | 0.829 ± 0.003 | 0.831 ± 0.004 | 0.705 ± 0.006 | 0.764 ± 0.008 | - | - |
| small | `eigh` | f64 | 8 | `16x16` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.034 ± 0.000 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `2x2` | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 8 | `32x32` | - | 0.123 ± 0.000 | 0.122 ± 0.000 | 0.114 ± 0.000 | 0.126 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `4x4` | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `8x8` | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `16x16` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.151 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.146 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `32x32` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.157 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.147 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.150 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | - | 0.313 ± 0.007 | 0.088 ± 0.000 | 0.156 ± 0.003 | 0.042 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | - | 0.236 ± 0.004 | 0.034 ± 0.000 | 0.122 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | - | 0.510 ± 0.010 | 0.240 ± 0.001 | 0.261 ± 0.005 | 0.132 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | - | 0.243 ± 0.003 | 0.038 ± 0.000 | 0.124 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | - | 0.261 ± 0.002 | 0.049 ± 0.000 | 0.132 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | - | 0.097 ± 0.001 | 0.086 ± 0.000 | 0.132 ± 0.002 | 0.043 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | - | 0.064 ± 0.001 | 0.033 ± 0.000 | 0.102 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | - | 0.194 ± 0.002 | 0.239 ± 0.000 | 0.226 ± 0.005 | 0.133 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | - | 0.067 ± 0.002 | 0.037 ± 0.000 | 0.105 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | - | 0.073 ± 0.001 | 0.048 ± 0.000 | 0.110 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | - | 0.946 ± 0.006 | 0.053 ± 0.000 | 0.230 ± 0.004 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | - | 0.896 ± 0.010 | 0.044 ± 0.000 | 0.211 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | - | 0.993 ± 0.021 | 0.081 ± 0.000 | 0.347 ± 0.003 | 0.053 ± 0.307 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | - | 0.918 ± 0.010 | 0.044 ± 0.000 | 0.210 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | - | 0.922 ± 0.018 | 0.046 ± 0.000 | 0.214 ± 0.003 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | - | 0.189 ± 0.002 | 0.050 ± 0.000 | 0.175 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | - | 0.173 ± 0.003 | 0.040 ± 0.000 | 0.160 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | - | 0.235 ± 0.004 | 0.079 ± 0.000 | 0.223 ± 0.005 | 0.122 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | - | 0.175 ± 0.004 | 0.041 ± 0.000 | 0.159 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | - | 0.178 ± 0.002 | 0.043 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.048 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.042 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.063 ± 0.002 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.043 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.045 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | - | 1.328 ± 0.028 | 0.079 ± 0.000 | 0.164 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | - | 1.258 ± 0.020 | 0.056 ± 0.000 | 0.145 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | - | 1.401 ± 0.027 | 0.141 ± 0.001 | 0.218 ± 0.004 | 0.076 ± 0.018 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | - | 1.309 ± 0.030 | 0.057 ± 0.000 | 0.145 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | - | 1.318 ± 0.013 | 0.062 ± 0.000 | 0.149 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | - | 0.168 ± 0.002 | 0.077 ± 0.000 | 0.166 ± 0.005 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | - | 0.149 ± 0.002 | 0.055 ± 0.000 | 0.147 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | - | 0.230 ± 0.004 | 0.139 ± 0.001 | 0.236 ± 0.004 | 0.091 ± 0.123 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | - | 0.150 ± 0.003 | 0.056 ± 0.000 | 0.147 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | - | 0.153 ± 0.002 | 0.061 ± 0.000 | 0.149 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.000 | 0.072 ± 0.000 | 0.023 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.067 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.075 ± 0.000 | 0.084 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.069 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.069 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | - | 0.175 ± 0.002 | 0.040 ± 0.000 | 0.334 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | - | 0.161 ± 0.005 | 0.034 ± 0.000 | 0.325 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | - | 0.188 ± 0.003 | 0.050 ± 0.000 | 0.354 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | - | 0.162 ± 0.003 | 0.034 ± 0.000 | 0.325 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | - | 0.166 ± 0.005 | 0.036 ± 0.000 | 0.327 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | - | 0.074 ± 0.001 | 0.039 ± 0.000 | 0.144 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | - | 0.068 ± 0.001 | 0.034 ± 0.000 | 0.138 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | - | 0.082 ± 0.002 | 0.049 ± 0.000 | 0.164 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | - | 0.069 ± 0.001 | 0.034 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | - | 0.072 ± 0.002 | 0.036 ± 0.000 | 0.139 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.000 | 0.108 ± 0.001 | 0.072 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.048 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.312 ± 0.000 | 0.272 ± 0.003 | 0.207 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.053 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.067 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | - | 0.321 ± 0.002 | 0.102 ± 0.000 | 0.257 ± 0.004 | 0.069 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | - | 0.193 ± 0.004 | 0.025 ± 0.000 | 0.185 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | - | 0.645 ± 0.012 | 0.314 ± 0.001 | 0.453 ± 0.003 | 0.204 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | - | 0.208 ± 0.005 | 0.030 ± 0.000 | 0.191 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | - | 0.231 ± 0.002 | 0.046 ± 0.000 | 0.208 ± 0.004 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | - | 0.118 ± 0.002 | 0.100 ± 0.000 | 0.174 ± 0.004 | 0.069 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | - | 0.061 ± 0.001 | 0.023 ± 0.000 | 0.111 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | - | 0.278 ± 0.003 | 0.313 ± 0.001 | 0.351 ± 0.009 | 0.204 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | - | 0.064 ± 0.001 | 0.029 ± 0.000 | 0.118 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | - | 0.075 ± 0.001 | 0.044 ± 0.000 | 0.130 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `32x32` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `64x64` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 9.323 ± 0.150 | 0.037 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `16x16` | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `32x32` | - | 0.040 ± 0.000 | 0.039 ± 0.001 | 0.036 ± 0.001 | 0.035 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=4` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=1` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.029 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=4` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.032 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `16x16` | - | 0.063 ± 0.000 | 0.061 ± 0.000 | 0.066 ± 0.000 | 0.073 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `2x2` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `32x32` | - | 0.214 ± 0.001 | 0.212 ± 0.001 | 0.221 ± 0.001 | 0.205 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `4x4` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `8x8` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 1.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`jax-cpu`, 1.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.0x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 40.8x faster than the slowest successful cell (`pytorch-cpu`, 9.364 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 33.8x faster than the slowest successful cell (`pytorch-cpu`, 2.620 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-trace`, 0.487 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 67.3x faster than the slowest successful cell (`pytorch-cpu`, 7.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 44.1x faster than the slowest successful cell (`pytorch-cpu`, 1.960 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 0.550 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 34.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 34.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`256x256`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 150.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`256x256`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 150.846 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 27.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 27.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 30.2x faster than the slowest successful cell (`jax-cpu`, 133.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-eager` is 20.0x faster than the slowest successful cell (`jax-cpu`, 132.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`jax-cpu`, 132.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-eager` is 20.2x faster than the slowest successful cell (`jax-cpu`, 146.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`jax-cpu`, 146.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-eager` is 14.3x faster than the slowest successful cell (`jax-cpu`, 127.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 127.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-eager` is 64.4x faster than the slowest successful cell (`pytorch-cpu`, 8.826 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-trace` is 70.5x faster than the slowest successful cell (`pytorch-cpu`, 8.826 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `jax-cpu` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 6.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-eager` is 31.3x faster than the slowest successful cell (`pytorch-cpu`, 6.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-trace` is 39.2x faster than the slowest successful cell (`pytorch-cpu`, 6.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-eager` is 64.9x faster than the slowest successful cell (`pytorch-cpu`, 23.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 72.7x faster than the slowest successful cell (`pytorch-cpu`, 23.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `jax-cpu` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 35.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-eager` is 79.9x faster than the slowest successful cell (`pytorch-cpu`, 35.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-trace` is 91.1x faster than the slowest successful cell (`pytorch-cpu`, 35.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-eager` is 45.9x faster than the slowest successful cell (`pytorch-cpu`, 30.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-trace` is 56.4x faster than the slowest successful cell (`pytorch-cpu`, 30.798 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-eager` is 91.6x faster than the slowest successful cell (`jax-cpu`, 131.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 102.4x faster than the slowest successful cell (`jax-cpu`, 131.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-eager` is 84.7x faster than the slowest successful cell (`jax-cpu`, 140.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-trace` is 94.3x faster than the slowest successful cell (`jax-cpu`, 140.936 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-eager` is 50.0x faster than the slowest successful cell (`jax-cpu`, 119.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-trace` is 59.2x faster than the slowest successful cell (`jax-cpu`, 119.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-eager` is 30.1x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-eager` is 29.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-eager` is 29.9x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 45.8x faster than the slowest successful cell (`tenferro-eager`, 0.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`tenferro-eager`, 0.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 63.9x faster than the slowest successful cell (`tenferro-eager`, 0.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 0.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`32x32`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 0.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 59.9x faster than the slowest successful cell (`tenferro-eager`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 41.6x faster than the slowest successful cell (`tenferro-eager`, 0.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 0.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 69.0x faster than the slowest successful cell (`tenferro-eager`, 1.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 1.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 92.8x faster than the slowest successful cell (`tenferro-eager`, 1.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 22.6x faster than the slowest successful cell (`tenferro-eager`, 1.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`32x32`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 1.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 83.4x faster than the slowest successful cell (`tenferro-eager`, 1.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`tenferro-eager`, 1.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 64.6x faster than the slowest successful cell (`tenferro-eager`, 1.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 1.318 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-eager`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`32x32,rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=8, shape=`64x64`): `jax-cpu` is 248.8x faster than the slowest successful cell (`pytorch-cpu`, 9.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=8, shape=`64x64`): `tenferro-eager` is 593.4x faster than the slowest successful cell (`pytorch-cpu`, 9.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=8, shape=`64x64`): `tenferro-trace` is 621.9x faster than the slowest successful cell (`pytorch-cpu`, 9.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

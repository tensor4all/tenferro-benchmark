# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_095941/run.yaml`
- Timestamp: `2026-10-31T00:59:41.343309+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261031_095941`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_095941/cpu_ops_t4_20261031_095941.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 34.218 ± 0.121 | 34.358 ± 0.043 | 26.935 ± 0.069 | 17.668 ± 0.181 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.540 ± 0.002 | 0.542 ± 0.002 | 0.408 ± 0.002 | 0.379 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 8.519 ± 0.008 | 8.640 ± 0.009 | 6.786 ± 0.016 | 3.497 ± 0.165 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 2.152 ± 0.002 | 2.135 ± 0.004 | 1.678 ± 0.005 | 1.122 ± 0.110 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.486 ± 0.001 | 0.482 ± 0.002 | 0.505 ± 0.002 | 0.425 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.128 ± 0.000 | 0.126 ± 0.000 | 0.133 ± 0.001 | 0.110 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.039 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.742 ± 0.005 | 2.746 ± 0.004 | 2.587 ± 0.013 | 1.686 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.051 ± 0.000 | 0.050 ± 0.000 | 0.048 ± 0.000 | 0.052 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.691 ± 0.001 | 0.696 ± 0.001 | 0.651 ± 0.003 | 0.654 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.179 ± 0.001 | 0.178 ± 0.000 | 0.169 ± 0.000 | 0.168 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 9.038 ± 0.013 | 9.021 ± 0.047 | 7.794 ± 0.019 | 3.897 ± 0.470 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.149 ± 0.000 | 0.148 ± 0.000 | 0.130 ± 0.001 | 0.145 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 2.260 ± 0.003 | 2.303 ± 0.006 | 1.962 ± 0.005 | 1.128 ± 0.300 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.570 ± 0.001 | 0.569 ± 0.001 | 0.496 ± 0.002 | 0.561 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.319 ± 0.004 | 0.340 ± 0.004 | 0.386 ± 0.015 | 1.513 ± 0.058 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.154 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.086 ± 0.001 | 0.084 ± 0.001 | 0.096 ± 0.004 | 0.289 ± 0.014 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.037 ± 0.002 | 0.161 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.040 ± 0.000 | 0.041 ± 0.000 | 0.043 ± 0.000 | 0.154 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.147 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.000 | 0.152 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.150 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.001 | 0.051 ± 0.000 | 0.066 ± 0.001 | 0.274 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.152 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.052 ± 0.001 | 0.157 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.024 ± 0.000 | 0.155 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.082 ± 0.001 | 0.083 ± 0.001 | 0.089 ± 0.010 | 0.456 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.154 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.036 ± 0.002 | 0.163 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.154 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 7.972 ± 0.041 | 7.860 ± 0.028 | 5.534 ± 0.047 | 4.341 ± 0.072 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.132 ± 0.001 | 0.132 ± 0.000 | 0.096 ± 0.000 | 0.134 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.011 ± 0.004 | 1.989 ± 0.004 | 1.414 ± 0.012 | 1.046 ± 0.055 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.506 ± 0.003 | 0.511 ± 0.001 | 0.365 ± 0.001 | 0.357 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.371 ± 0.001 | 0.375 ± 0.001 | 0.209 ± 0.001 | 0.324 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.099 ± 0.000 | 0.100 ± 0.000 | 0.057 ± 0.001 | 0.087 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.000 | 0.031 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.009 ± 0.002 | 1.001 ± 0.004 | 0.631 ± 0.003 | 0.943 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.263 ± 0.000 | 0.256 ± 0.001 | 0.164 ± 0.001 | 0.241 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.070 ± 0.000 | 0.070 ± 0.000 | 0.046 ± 0.000 | 0.064 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 2.657 ± 0.004 | 2.643 ± 0.003 | 1.633 ± 0.006 | 1.245 ± 0.109 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.033 ± 0.000 | 0.048 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.678 ± 0.004 | 0.671 ± 0.001 | 0.421 ± 0.002 | 0.679 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.174 ± 0.000 | 0.173 ± 0.000 | 0.111 ± 0.000 | 0.176 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.610 ± 0.005 | 2.598 ± 0.010 | 0.905 ± 0.011 | 2.127 ± 0.028 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.051 ± 0.000 | 0.052 ± 0.000 | 0.036 ± 0.000 | 0.059 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.660 ± 0.002 | 0.658 ± 0.005 | 0.251 ± 0.002 | 0.553 ± 0.031 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.176 ± 0.000 | 0.176 ± 0.001 | 0.082 ± 0.001 | 0.190 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.315 ± 0.009 | 0.280 ± 0.001 | 0.085 ± 0.001 | 0.336 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.086 ± 0.000 | 0.079 ± 0.000 | 0.037 ± 0.001 | 0.091 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.029 ± 0.000 | 0.028 ± 0.000 | 0.024 ± 0.000 | 0.025 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.486 ± 0.002 | 0.452 ± 0.001 | 0.121 ± 0.002 | 0.485 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.129 ± 0.000 | 0.122 ± 0.000 | 0.045 ± 0.001 | 0.128 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.026 ± 0.000 | 0.039 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.975 ± 0.003 | 0.953 ± 0.626 | 0.340 ± 0.004 | 0.920 ± 0.018 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.025 ± 0.000 | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.254 ± 0.001 | 0.247 ± 0.000 | 0.108 ± 0.001 | 0.255 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.071 ± 0.000 | 0.070 ± 0.000 | 0.042 ± 0.000 | 0.073 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 58.659 ± 0.319 | 58.561 ± 0.215 | 59.964 ± 0.285 | 30.104 ± 0.153 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.929 ± 0.004 | 0.921 ± 0.003 | 0.920 ± 0.002 | 0.589 ± 0.071 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 14.730 ± 0.102 | 14.719 ± 0.029 | 14.978 ± 0.036 | 5.750 ± 0.381 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 3.690 ± 0.019 | 3.693 ± 0.013 | 3.737 ± 0.009 | 1.928 ± 0.204 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.722 ± 0.008 | 1.712 ± 0.008 | 1.329 ± 0.006 | 1.552 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.030 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.433 ± 0.002 | 0.434 ± 0.002 | 0.340 ± 0.001 | 0.390 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.116 ± 0.000 | 0.115 ± 0.000 | 0.092 ± 0.000 | 0.108 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 5.075 ± 0.007 | 5.050 ± 0.022 | 5.347 ± 0.016 | 2.297 ± 0.311 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.089 ± 0.000 | 0.088 ± 0.001 | 0.093 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 1.279 ± 0.007 | 1.275 ± 0.002 | 1.346 ± 0.002 | 1.271 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.325 ± 0.001 | 0.324 ± 0.001 | 0.342 ± 0.002 | 0.322 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 15.859 ± 0.077 | 15.824 ± 0.014 | 18.269 ± 0.035 | 4.451 ± 2.739 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.258 ± 0.001 | 0.255 ± 0.001 | 0.293 ± 0.001 | 0.258 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 3.980 ± 0.010 | 3.966 ± 0.011 | 4.590 ± 0.007 | 2.057 ± 0.119 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 1.000 ± 0.002 | 0.998 ± 0.001 | 1.150 ± 0.006 | 0.700 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.813 ± 0.003 | 9.433 ± 0.075 | 3.219 ± 0.046 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.257 ± 0.002 | 0.054 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.470 ± 0.001 | 2.477 ± 0.021 | 0.784 ± 0.012 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.123 ± 0.000 | 0.699 ± 0.014 | 0.196 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.257 ± 0.003 | 0.195 ± 0.001 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.107 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.077 ± 0.000 | 0.126 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.111 ± 0.002 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.332 ± 0.001 | 0.269 ± 0.001 | 0.795 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.111 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.095 ± 0.000 | 0.221 ± 0.001 | 0.205 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.133 ± 0.002 | 0.057 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.511 ± 0.003 | 6.889 ± 0.040 | 1.343 ± 0.018 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.218 ± 0.002 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.133 ± 0.002 | 1.847 ± 0.010 | 0.323 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.543 ± 0.003 | 0.087 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.215 ± 0.008 | 1.863 ± 0.046 | 3.995 ± 0.682 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.102 ± 0.000 | 0.125 ± 0.001 | 0.097 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.087 ± 0.002 | 0.571 ± 0.006 | 1.084 ± 0.049 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.307 ± 0.001 | 0.269 ± 0.004 | 0.338 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.613 ± 0.001 | 0.208 ± 0.001 | 0.702 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.077 ± 0.001 | 0.028 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.180 ± 0.000 | 0.110 ± 0.001 | 0.186 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.071 ± 0.000 | 0.085 ± 0.001 | 0.054 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.878 ± 0.003 | 0.340 ± 0.008 | 1.005 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.080 ± 0.001 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.245 ± 0.000 | 0.141 ± 0.001 | 0.263 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.087 ± 0.000 | 0.093 ± 0.002 | 0.077 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.684 ± 0.007 | 0.864 ± 0.004 | 1.702 ± 0.105 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.060 ± 0.000 | 0.093 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.455 ± 0.001 | 0.350 ± 0.004 | 0.468 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.139 ± 0.000 | 0.144 ± 0.002 | 0.131 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 160.029 ± 0.382 | 158.139 ± 0.621 | 230.962 ± 45.338 | 205.499 ± 50.571 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 1.916 ± 0.002 | 10.739 ± 0.013 | 7.446 ± 64.009 | 4.612 ± 36.121 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 17.533 ± 0.023 | 17.333 ± 0.027 | 135.422 ± 129.206 | 13.327 ± 127.877 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 41.762 ± 0.079 | 41.863 ± 0.097 | 42.805 ± 18.067 | 138.420 ± 75.049 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.438 ± 0.000 | 0.438 ± 0.002 | 0.286 ± 0.001 | 0.460 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 218.750 ± 28.562 | 202.799 ± 1.748 | 254.797 ± 7.844 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 21.337 ± 3.995 | 6.068 ± 0.058 | 11.115 ± 0.748 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 50.997 ± 0.009 | 34.421 ± 0.221 | 50.758 ± 3.351 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 221.998 ± 31.840 | 140.038 ± 2.217 | 252.164 ± 6.496 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 36.000 ± 5.008 | 5.253 ± 0.022 | 11.018 ± 0.538 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 50.999 ± 0.009 | 25.018 ± 0.093 | 51.596 ± 2.669 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 166.996 ± 2.020 | 241.185 ± 5.199 | 178.450 ± 18.264 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 57.010 ± 31.001 | 5.603 ± 0.091 | 4.836 ± 1.239 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 56.069 ± 32.001 | 43.361 ± 0.173 | 29.992 ± 3.189 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 162.363 ± 6.890 | 115.886 ± 0.853 | 173.572 ± 17.242 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 36.998 ± 27.337 | 2.687 ± 0.036 | 4.492 ± 1.569 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 77.025 ± 2.032 | 18.473 ± 0.486 | 30.098 ± 5.089 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 33.992 ± 0.006 | 32.996 ± 2.125 | 27.458 ± 0.200 | 20.319 ± 0.138 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.090 ± 0.001 | 0.060 ± 0.000 | 0.085 ± 0.004 | 0.141 ± 0.017 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 13.997 ± 0.009 | 4.497 ± 1.496 | 0.432 ± 0.004 | 0.591 ± 0.057 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 14.014 ± 7.000 | 9.017 ± 4.998 | 5.070 ± 0.047 | 3.336 ± 0.620 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.039 ± 0.000 | 0.026 ± 0.000 | 0.026 ± 0.001 | 0.035 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 72.974 ± 1.588 | 76.211 ± 2.219 | 59.689 ± 1.310 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.198 ± 0.001 | 0.281 ± 0.003 | 0.275 ± 0.012 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 11.999 ± 0.009 | 1.319 ± 0.024 | 1.377 ± 0.093 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 18.999 ± 1.008 | 12.168 ± 0.071 | 8.526 ± 0.676 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.000 | 0.099 ± 0.002 | 0.082 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 198.282 ± 43.369 | 163.496 ± 4.116 | 261.279 ± 14.294 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 58.991 ± 53.006 | 4.245 ± 0.031 | 8.255 ± 1.808 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 123.001 ± 20.005 | 28.891 ± 0.079 | 46.081 ± 4.783 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 210.987 ± 48.024 | 158.121 ± 0.617 | 247.014 ± 10.014 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 56.986 ± 39.981 | 4.519 ± 0.038 | 8.710 ± 1.279 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 134.997 ± 7.973 | 31.387 ± 0.340 | 46.304 ± 5.287 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 30.994 ± 0.042 | 44.606 ± 46.012 | 27.473 ± 1.087 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 10.604 ± 12.397 | 0.262 ± 0.002 | 0.213 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 25.998 ± 0.019 | 27.913 ± 3.993 | 1.636 ± 2.119 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 13.181 ± 0.018 | 6.716 ± 0.055 | 132.514 ± 131.866 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.077 ± 0.000 | 0.113 ± 0.001 | 0.066 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 24.282 ± 0.051 | 12.857 ± 0.072 | 20.179 ± 1.344 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 17.000 ± 0.002 | 0.771 ± 0.041 | 0.771 ± 0.013 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 12.987 ± 0.021 | 2.872 ± 0.151 | 4.214 ± 0.027 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 31.999 ± 0.003 | 22.395 ± 0.209 | 19.413 ± 0.285 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 17.000 ± 2.996 | 0.869 ± 0.023 | 0.782 ± 0.007 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 13.070 ± 0.021 | 4.326 ± 0.054 | 4.289 ± 0.060 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 443.966 ± 11.731 | 390.357 ± 3.347 | 522.189 ± 7.203 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 26.993 ± 0.018 | 3.275 ± 0.021 | 4.135 ± 0.163 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 39.747 ± 2.998 | 15.267 ± 0.053 | 18.504 ± 1.225 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 107.987 ± 0.978 | 90.706 ± 1.122 | 108.651 ± 4.828 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.754 ± 0.003 | 0.820 ± 0.010 | 0.805 ± 0.007 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 412.961 ± 10.422 | 436.993 ± 6.326 | 523.162 ± 12.414 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 25.100 ± 0.048 | 16.468 ± 0.072 | 18.911 ± 0.697 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 95.484 ± 1.459 | 85.515 ± 0.320 | 109.892 ± 2.768 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 431.022 ± 2.908 | 351.521 ± 1.977 | 524.287 ± 5.736 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 39.002 ± 3.001 | 15.331 ± 0.053 | 18.535 ± 0.775 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 105.588 ± 2.698 | 82.218 ± 4.902 | 107.110 ± 4.054 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 20.207 ± 0.026 | 20.158 ± 0.029 | 22.363 ± 0.116 | 19.655 ± 0.575 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.055 ± 0.000 | 0.056 ± 0.000 | 0.096 ± 0.003 | 0.116 ± 0.012 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.347 ± 0.001 | 0.347 ± 0.001 | 0.453 ± 0.007 | 0.597 ± 0.065 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 2.601 ± 0.013 | 2.572 ± 0.003 | 3.709 ± 0.024 | 2.895 ± 0.708 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 5.085 ± 0.007 | 5.058 ± 0.010 | 5.853 ± 0.014 | 5.431 ± 0.270 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.330 ± 0.008 | 1.323 ± 0.004 | 1.551 ± 0.022 | 1.614 ± 0.374 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 84.111 ± 0.209 | 82.341 ± 0.195 | 66.755 ± 0.537 | 86.873 ± 2.538 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.889 ± 0.001 | 10.994 ± 0.008 | 0.491 ± 0.002 | 0.844 ± 0.006 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 13.179 ± 0.011 | 13.030 ± 0.006 | 2.133 ± 0.009 | 3.229 ± 0.066 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 25.297 ± 0.032 | 25.231 ± 0.056 | 12.933 ± 0.042 | 17.967 ± 0.106 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.121 ± 0.000 | 0.119 ± 0.000 | 0.142 ± 0.001 | 0.118 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 12.705 ± 0.047 | 23.048 ± 0.038 | 35.923 ± 52.443 | 137.515 ± 110.822 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 13.507 ± 0.071 | 23.838 ± 0.037 | 48.978 ± 6.010 | 131.985 ± 99.403 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 14.716 ± 0.077 | 25.074 ± 0.088 | 55.448 ± 40.145 | 45.990 ± 80.350 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.143 ± 0.000 | 10.517 ± 0.012 | 0.176 ± 0.001 | 0.179 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.160 ± 0.003 | 10.543 ± 0.007 | 2.260 ± 0.007 | 0.284 ± 1.796 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.201 ± 0.003 | 10.568 ± 0.002 | 8.306 ± 0.032 | 0.331 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.487 ± 0.001 | 14.005 ± 0.409 | 21.471 ± 2.496 | 19.950 ± 19.443 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.546 ± 0.006 | 13.997 ± 0.025 | 32.703 ± 1.015 | 4.131 ± 5.085 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.667 ± 0.004 | 15.002 ± 0.017 | 29.896 ± 2.991 | 2.736 ± 11.236 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 2.463 ± 0.023 | 13.786 ± 2.983 | 2.083 ± 19.834 | 11.214 ± 4.712 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 2.653 ± 0.010 | 16.002 ± 0.054 | 8.292 ± 2.986 | 36.954 ± 116.825 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 3.026 ± 0.010 | 13.335 ± 1.024 | 4.125 ± 0.038 | 13.333 ± 99.579 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.039 ± 0.001 | 0.043 ± 0.000 | 0.052 ± 0.001 | 0.049 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.047 ± 0.000 | 0.050 ± 0.001 | 0.064 ± 0.001 | 0.101 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.061 ± 0.001 | 0.066 ± 0.001 | 0.086 ± 0.002 | 0.137 ± 0.007 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 375.983 ± 8.453 | 375.407 ± 8.724 | 326.935 ± 2.722 | 416.228 ± 8.922 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 3.726 ± 8.158 | 13.423 ± 0.033 | 2.900 ± 0.020 | 3.319 ± 0.029 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 24.977 ± 0.089 | 24.491 ± 0.058 | 14.123 ± 0.034 | 15.525 ± 0.120 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 88.437 ± 0.579 | 88.358 ± 0.396 | 63.632 ± 0.111 | 91.739 ± 0.395 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.708 ± 0.005 | 0.707 ± 0.002 | 0.705 ± 0.004 | 0.756 ± 0.012 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.042 ± 0.000 | 0.041 ± 0.000 | 0.033 ± 0.000 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.127 ± 0.000 | 0.126 ± 0.000 | 0.110 ± 0.000 | 0.125 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.150 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.144 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.153 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.147 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.151 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.068 ± 0.000 | 0.160 ± 0.005 | 0.042 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.033 ± 0.000 | 0.123 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.158 ± 0.000 | 0.259 ± 0.005 | 0.131 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.035 ± 0.000 | 0.126 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.042 ± 0.000 | 0.136 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.067 ± 0.000 | 0.135 ± 0.002 | 0.043 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.031 ± 0.000 | 0.106 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.157 ± 0.001 | 0.222 ± 0.004 | 0.133 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.034 ± 0.000 | 0.108 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.040 ± 0.000 | 0.115 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.046 ± 0.000 | 0.228 ± 0.004 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.040 ± 0.000 | 0.211 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.063 ± 0.000 | 0.281 ± 0.003 | 0.059 ± 0.019 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.040 ± 0.000 | 0.208 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.042 ± 0.000 | 0.213 ± 0.003 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.044 ± 0.000 | 0.182 ± 0.012 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.037 ± 0.000 | 0.163 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.065 ± 0.000 | 0.214 ± 0.005 | 0.054 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.037 ± 0.000 | 0.163 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.039 ± 0.000 | 0.169 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.047 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.042 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.061 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.068 ± 0.000 | 0.165 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.057 ± 0.000 | 0.143 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.099 ± 0.000 | 0.205 ± 0.005 | 0.076 ± 0.028 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.058 ± 0.000 | 0.142 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.060 ± 0.000 | 0.151 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.067 ± 0.000 | 0.167 ± 0.007 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.056 ± 0.000 | 0.151 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.100 ± 0.000 | 0.209 ± 0.006 | 0.068 ± 0.016 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.057 ± 0.000 | 0.151 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.060 ± 0.000 | 0.154 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.070 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.064 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.081 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.067 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.067 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.337 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.026 ± 0.000 | 0.327 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.036 ± 0.000 | 0.354 ± 0.008 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.027 ± 0.000 | 0.326 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.028 ± 0.000 | 0.329 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.029 ± 0.000 | 0.148 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.026 ± 0.000 | 0.140 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.036 ± 0.000 | 0.167 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.026 ± 0.000 | 0.141 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.001 | 0.108 ± 0.001 | 0.071 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.047 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.213 ± 0.001 | 0.266 ± 0.002 | 0.207 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.053 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.067 ± 0.001 | 0.023 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.081 ± 0.000 | 0.264 ± 0.004 | 0.069 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.188 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.216 ± 0.000 | 0.452 ± 0.004 | 0.205 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.026 ± 0.000 | 0.194 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.037 ± 0.000 | 0.211 ± 0.005 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.079 ± 0.000 | 0.178 ± 0.005 | 0.069 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.021 ± 0.000 | 0.115 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.214 ± 0.000 | 0.351 ± 0.011 | 0.205 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.024 ± 0.000 | 0.121 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.035 ± 0.000 | 0.136 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.027 ± 0.000 | 0.034 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.032 ± 0.000 | 0.033 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.030 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.067 ± 0.000 | 0.066 ± 0.000 | 0.065 ± 0.000 | 0.073 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.206 ± 0.000 | 0.204 ± 0.001 | 0.216 ± 0.001 | 0.203 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 28.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `tenferro-eager` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch256 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 24.7x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 6.889 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 1.847 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 135.422 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 57.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 57.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`tenferro-trace`, 36.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`256x256`): `jax-cpu` is 23.7x faster than the slowest successful cell (`tenferro-eager`, 13.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 32.4x faster than the slowest successful cell (`tenferro-eager`, 13.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`tenferro-trace`, 58.991 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 56.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`128x128,rhs=1`): `jax-cpu` is 49.8x faster than the slowest successful cell (`tenferro-trace`, 10.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`128x128,rhs=1`): `pytorch-cpu` is 40.4x faster than the slowest successful cell (`tenferro-trace`, 10.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 27.913 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 19.7x faster than the slowest successful cell (`jax-cpu`, 132.514 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 132.514 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 22.0x faster than the slowest successful cell (`tenferro-trace`, 17.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 22.0x faster than the slowest successful cell (`tenferro-trace`, 17.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 21.8x faster than the slowest successful cell (`tenferro-trace`, 17.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 17.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=4, shape=`128x128`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 10.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=4, shape=`128x128`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`tenferro-trace`, 10.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 10.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=1`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 137.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=1`): `jax-cpu` is 58.6x faster than the slowest successful cell (`tenferro-trace`, 10.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=1`): `pytorch-cpu` is 59.8x faster than the slowest successful cell (`tenferro-trace`, 10.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=1`): `tenferro-eager` is 73.4x faster than the slowest successful cell (`tenferro-trace`, 10.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `jax-cpu` is 37.2x faster than the slowest successful cell (`tenferro-trace`, 10.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-eager` is 66.0x faster than the slowest successful cell (`tenferro-trace`, 10.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `jax-cpu` is 31.9x faster than the slowest successful cell (`tenferro-trace`, 10.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-eager` is 52.6x faster than the slowest successful cell (`tenferro-trace`, 10.568 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-eager` is 44.1x faster than the slowest successful cell (`pytorch-cpu`, 21.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-eager` is 59.8x faster than the slowest successful cell (`pytorch-cpu`, 32.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 29.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-eager` is 44.8x faster than the slowest successful cell (`pytorch-cpu`, 29.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-eager` is 13.9x faster than the slowest successful cell (`jax-cpu`, 36.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 28.9x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 32.4x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.326 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.326 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

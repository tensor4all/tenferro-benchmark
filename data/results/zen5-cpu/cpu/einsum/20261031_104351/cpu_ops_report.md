# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_104351/run.yaml`
- Timestamp: `2026-10-31T01:43:51.562747+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261031_104351`.

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_104351/cpu_ops_t8_20261031_104351.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 34.494 ± 0.082 | 34.562 ± 0.066 | 27.256 ± 0.078 | 7.949 ± 2.926 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.551 ± 0.001 | 0.543 ± 0.001 | 0.409 ± 0.002 | 0.392 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 8.665 ± 0.010 | 8.574 ± 0.019 | 6.881 ± 0.030 | 2.720 ± 0.153 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 2.181 ± 0.006 | 2.159 ± 0.002 | 1.692 ± 0.006 | 0.580 ± 0.024 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.487 ± 0.002 | 0.491 ± 0.001 | 0.507 ± 0.003 | 0.432 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.129 ± 0.000 | 0.128 ± 0.000 | 0.133 ± 0.001 | 0.113 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.038 ± 0.000 | 0.039 ± 0.000 | 0.032 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 2.778 ± 0.005 | 2.785 ± 0.003 | 2.613 ± 0.007 | 1.738 ± 0.024 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.052 ± 0.000 | 0.051 ± 0.000 | 0.048 ± 0.000 | 0.052 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.698 ± 0.001 | 0.699 ± 0.001 | 0.658 ± 0.004 | 0.669 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.181 ± 0.000 | 0.181 ± 0.000 | 0.170 ± 0.000 | 0.170 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 9.078 ± 0.030 | 9.058 ± 0.014 | 7.889 ± 0.026 | 3.631 ± 0.054 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.150 ± 0.000 | 0.150 ± 0.000 | 0.130 ± 0.000 | 0.146 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 2.296 ± 0.005 | 2.281 ± 0.003 | 1.993 ± 0.007 | 0.868 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.575 ± 0.001 | 0.576 ± 0.001 | 0.500 ± 0.001 | 0.568 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 0.349 ± 0.006 | 0.343 ± 0.004 | 0.257 ± 0.009 | 1.502 ± 0.033 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.012 ± 0.001 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.156 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.094 ± 0.001 | 0.086 ± 0.000 | 0.080 ± 0.003 | 0.325 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.029 ± 0.000 | 0.025 ± 0.000 | 0.037 ± 0.002 | 0.167 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.044 ± 0.000 | 0.041 ± 0.000 | 0.045 ± 0.001 | 0.155 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.146 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.000 | 0.154 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.152 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.070 ± 0.001 | 0.050 ± 0.001 | 0.071 ± 0.001 | 0.280 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.151 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.022 ± 0.001 | 0.017 ± 0.000 | 0.053 ± 0.001 | 0.157 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.025 ± 0.000 | 0.152 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.088 ± 0.001 | 0.085 ± 0.001 | 0.075 ± 0.002 | 0.486 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.153 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.033 ± 0.001 | 0.168 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.156 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 7.963 ± 0.014 | 7.966 ± 0.028 | 5.589 ± 0.021 | 2.618 ± 0.086 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.135 ± 0.000 | 0.133 ± 0.000 | 0.099 ± 0.000 | 0.135 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 1.994 ± 0.002 | 1.985 ± 0.003 | 1.437 ± 0.013 | 0.564 ± 0.012 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.515 ± 0.002 | 0.521 ± 0.003 | 0.373 ± 0.005 | 0.372 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.370 ± 0.001 | 0.369 ± 0.001 | 0.205 ± 0.001 | 0.329 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.100 ± 0.000 | 0.098 ± 0.015 | 0.057 ± 0.001 | 0.086 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.000 | 0.030 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 1.011 ± 0.002 | 1.010 ± 0.002 | 0.635 ± 0.004 | 0.956 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.258 ± 0.002 | 0.257 ± 0.001 | 0.164 ± 0.001 | 0.244 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.071 ± 0.000 | 0.070 ± 0.000 | 0.047 ± 0.000 | 0.066 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 2.671 ± 0.010 | 2.658 ± 0.012 | 1.640 ± 0.006 | 0.999 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.035 ± 0.001 | 0.049 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.675 ± 0.001 | 0.676 ± 0.002 | 0.428 ± 0.002 | 0.689 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.175 ± 0.000 | 0.174 ± 0.000 | 0.113 ± 0.001 | 0.178 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.647 ± 0.006 | 2.618 ± 0.022 | 0.757 ± 0.015 | 2.035 ± 0.078 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.052 ± 0.000 | 0.052 ± 0.000 | 0.036 ± 0.001 | 0.061 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.681 ± 0.005 | 0.668 ± 0.001 | 0.229 ± 0.004 | 0.526 ± 0.022 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.180 ± 0.000 | 0.178 ± 0.001 | 0.079 ± 0.004 | 0.205 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.334 ± 0.002 | 0.291 ± 0.000 | 0.083 ± 0.001 | 0.357 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.091 ± 0.000 | 0.082 ± 0.000 | 0.037 ± 0.001 | 0.095 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.025 ± 0.000 | 0.029 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.495 ± 0.001 | 0.463 ± 0.001 | 0.118 ± 0.001 | 0.508 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.131 ± 0.000 | 0.124 ± 0.000 | 0.045 ± 0.000 | 0.135 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.027 ± 0.001 | 0.040 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.990 ± 0.001 | 0.962 ± 0.001 | 0.319 ± 0.003 | 0.962 ± 0.029 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.001 | 0.022 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.259 ± 0.000 | 0.251 ± 0.001 | 0.105 ± 0.002 | 0.265 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.072 ± 0.000 | 0.071 ± 0.000 | 0.042 ± 0.001 | 0.076 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 58.821 ± 0.137 | 59.449 ± 0.040 | 60.184 ± 0.113 | 15.375 ± 3.342 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.933 ± 0.002 | 0.937 ± 0.003 | 0.919 ± 0.002 | 0.382 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 14.924 ± 0.095 | 14.878 ± 0.031 | 15.145 ± 0.035 | 3.905 ± 0.888 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 3.683 ± 0.011 | 3.736 ± 0.006 | 3.763 ± 0.007 | 0.995 ± 0.060 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 1.713 ± 0.014 | 1.726 ± 0.009 | 1.336 ± 0.006 | 1.554 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.030 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.437 ± 0.002 | 0.442 ± 0.003 | 0.341 ± 0.001 | 0.389 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.117 ± 0.001 | 0.116 ± 0.000 | 0.093 ± 0.000 | 0.109 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 5.105 ± 0.006 | 5.103 ± 0.009 | 5.398 ± 0.012 | 1.658 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.090 ± 0.000 | 0.089 ± 0.000 | 0.093 ± 0.000 | 0.092 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 1.280 ± 0.003 | 1.280 ± 0.002 | 1.357 ± 0.006 | 1.282 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.327 ± 0.001 | 0.327 ± 0.001 | 0.346 ± 0.000 | 0.324 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 15.930 ± 0.026 | 15.961 ± 0.030 | 18.395 ± 0.067 | 3.213 ± 0.981 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.258 ± 0.001 | 0.257 ± 0.000 | 0.293 ± 0.001 | 0.258 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 3.986 ± 0.006 | 3.995 ± 0.005 | 4.624 ± 0.013 | 0.829 ± 0.058 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 1.004 ± 0.002 | 1.003 ± 0.002 | 1.157 ± 0.003 | 0.711 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.912 ± 0.010 | 9.505 ± 0.039 | 3.402 ± 0.028 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.260 ± 0.002 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.479 ± 0.001 | 2.579 ± 0.038 | 0.813 ± 0.010 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.127 ± 0.001 | 0.712 ± 0.009 | 0.206 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.266 ± 0.000 | 0.197 ± 0.001 | 0.026 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.107 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.000 | 0.127 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.111 ± 0.001 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.339 ± 0.001 | 0.277 ± 0.002 | 0.825 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.112 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.098 ± 0.000 | 0.223 ± 0.001 | 0.210 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.134 ± 0.001 | 0.058 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.523 ± 0.002 | 7.015 ± 0.034 | 1.327 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.219 ± 0.001 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.140 ± 0.002 | 1.890 ± 0.021 | 0.332 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.544 ± 0.023 | 0.089 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.229 ± 0.014 | 1.698 ± 0.034 | 4.044 ± 0.181 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.104 ± 0.000 | 0.124 ± 0.002 | 0.098 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.099 ± 0.004 | 0.565 ± 0.023 | 1.044 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.313 ± 0.001 | 0.270 ± 0.008 | 0.359 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.636 ± 0.001 | 0.211 ± 0.003 | 0.736 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.000 | 0.078 ± 0.000 | 0.029 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.186 ± 0.000 | 0.110 ± 0.002 | 0.194 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.000 | 0.086 ± 0.000 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.900 ± 0.001 | 0.340 ± 0.007 | 1.050 ± 0.036 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.081 ± 0.001 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.251 ± 0.000 | 0.142 ± 0.002 | 0.271 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.090 ± 0.000 | 0.094 ± 0.001 | 0.080 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.707 ± 0.005 | 0.862 ± 0.005 | 1.852 ± 0.032 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.063 ± 0.000 | 0.095 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.465 ± 0.001 | 0.345 ± 0.008 | 0.485 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.141 ± 0.000 | 0.144 ± 0.002 | 0.134 ± 0.002 | - | - |
| large | `eigh` | f64 | 8 | `1024x1024` | - | 134.867 ± 2.091 | 134.837 ± 3.054 | 218.611 ± 44.000 | 259.882 ± 34.054 | - | - |
| large | `eigh` | f64 | 8 | `128x128` | - | 10.805 ± 0.007 | 1.792 ± 0.004 | 76.238 ± 8.496 | 3.253 ± 5.912 | - | - |
| large | `eigh` | f64 | 8 | `256x256` | - | 23.923 ± 6.602 | 18.023 ± 5.827 | 134.925 ± 17.036 | 14.443 ± 92.729 | - | - |
| large | `eigh` | f64 | 8 | `512x512` | - | 43.108 ± 9.824 | 39.633 ± 0.039 | 159.427 ± 31.618 | 159.341 ± 36.561 | - | - |
| large | `eigh` | f64 | 8 | `64x64` | - | 0.444 ± 0.001 | 0.443 ± 0.001 | 0.281 ± 0.001 | 0.467 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | - | - | 189.998 ± 35.440 | 133.390 ± 0.579 | 206.114 ± 17.387 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | - | - | 53.968 ± 5.998 | 4.842 ± 0.048 | 11.816 ± 1.752 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | - | - | 89.992 ± 15.211 | 21.225 ± 0.166 | 46.909 ± 5.578 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | - | - | 188.997 ± 6.165 | 96.324 ± 0.423 | 200.827 ± 12.909 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | - | - | 58.999 ± 9.000 | 4.358 ± 0.038 | 11.510 ± 0.875 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | - | - | 92.996 ± 24.982 | 16.950 ± 0.126 | 47.014 ± 3.511 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | - | - | 128.989 ± 30.015 | 146.653 ± 1.455 | 121.086 ± 7.952 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | - | - | 74.000 ± 12.142 | 4.113 ± 0.042 | 6.967 ± 3.623 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | - | - | 88.999 ± 27.714 | 27.335 ± 0.249 | 23.690 ± 6.726 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | - | - | 127.206 ± 20.827 | 72.352 ± 0.181 | 120.879 ± 18.902 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | - | - | 53.034 ± 24.660 | 2.120 ± 0.059 | 6.375 ± 3.009 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | - | - | 85.980 ± 18.445 | 11.595 ± 0.084 | 22.187 ± 8.472 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `1024x1024` | - | 32.030 ± 3.981 | 24.006 ± 3.004 | 14.903 ± 0.134 | 12.381 ± 1.218 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `128x128` | - | 0.079 ± 0.001 | 0.042 ± 0.000 | 0.061 ± 0.001 | 0.105 ± 0.002 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `256x256` | - | 3.084 ± 2.996 | 13.999 ± 2.001 | 0.269 ± 0.004 | 0.375 ± 0.015 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `512x512` | - | 11.996 ± 7.990 | 14.999 ± 0.019 | 2.868 ± 0.016 | 2.051 ± 0.087 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `64x64` | - | 0.040 ± 0.000 | 0.026 ± 0.000 | 0.023 ± 0.002 | 0.039 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 46.999 ± 0.004 | 43.315 ± 0.278 | 33.339 ± 1.525 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.152 ± 0.001 | 0.198 ± 0.002 | 0.219 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 11.999 ± 0.001 | 0.792 ± 0.004 | 0.813 ± 0.010 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 20.996 ± 1.000 | 6.929 ± 0.034 | 5.015 ± 0.203 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.000 | 0.089 ± 0.002 | 0.085 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | - | - | 211.007 ± 25.983 | 110.665 ± 0.384 | 195.889 ± 20.533 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | - | - | 95.998 ± 19.002 | 3.438 ± 0.030 | 9.597 ± 2.868 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | - | - | 138.003 ± 10.990 | 20.611 ± 0.147 | 39.287 ± 6.327 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | - | - | 227.997 ± 37.002 | 105.321 ± 0.507 | 186.900 ± 14.890 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | - | - | 68.998 ± 18.999 | 3.672 ± 0.030 | 8.770 ± 3.378 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | - | - | 118.985 ± 18.993 | 22.392 ± 0.182 | 43.959 ± 12.119 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 36.998 ± 2.862 | 123.900 ± 80.048 | 134.884 ± 17.327 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 10.717 ± 0.011 | 0.271 ± 0.002 | 0.225 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 26.000 ± 2.010 | 29.816 ± 6.963 | 1.681 ± 32.610 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 12.567 ± 0.086 | 94.772 ± 9.030 | 134.502 ± 160.711 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.078 ± 0.000 | 0.116 ± 0.001 | 0.071 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | - | - | 19.183 ± 0.063 | 9.849 ± 0.056 | 15.058 ± 0.388 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | - | - | 19.999 ± 0.011 | 0.791 ± 0.013 | 0.576 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | - | - | 12.404 ± 0.021 | 2.027 ± 0.045 | 3.678 ± 0.048 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | - | - | 26.999 ± 3.000 | 14.977 ± 0.102 | 14.489 ± 1.408 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | - | - | 20.000 ± 0.006 | 0.896 ± 0.029 | 0.589 ± 0.012 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | - | - | 12.387 ± 0.261 | 3.029 ± 0.019 | 3.665 ± 0.183 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 372.068 ± 15.603 | 309.701 ± 11.905 | 464.895 ± 10.959 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 27.008 ± 0.349 | 3.046 ± 0.016 | 4.147 ± 0.112 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 38.995 ± 0.029 | 14.091 ± 0.096 | 19.060 ± 1.260 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 99.002 ± 4.831 | 70.463 ± 0.189 | 101.616 ± 5.582 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.757 ± 0.003 | 0.806 ± 0.005 | 0.801 ± 0.010 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | - | - | 341.597 ± 8.461 | 328.967 ± 1.306 | 470.148 ± 23.539 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | - | - | 36.632 ± 6.658 | 15.159 ± 0.050 | 19.293 ± 1.919 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | - | - | 85.046 ± 0.183 | 68.653 ± 0.181 | 102.943 ± 2.980 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | - | - | 351.400 ± 6.011 | 278.482 ± 0.691 | 465.000 ± 15.933 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | - | - | 43.139 ± 9.722 | 14.187 ± 0.028 | 19.654 ± 0.961 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | - | - | 98.002 ± 1.015 | 64.073 ± 0.115 | 104.326 ± 6.176 | - | - |
| large | `matmul` | f64 | 8 | `1024x1024` | - | 10.520 ± 0.015 | 10.638 ± 0.022 | 12.784 ± 0.079 | 11.503 ± 0.635 | - | - |
| large | `matmul` | f64 | 8 | `128x128` | - | 0.040 ± 0.001 | 0.038 ± 0.000 | 0.068 ± 0.002 | 0.078 ± 0.005 | - | - |
| large | `matmul` | f64 | 8 | `256x256` | - | 0.197 ± 0.001 | 0.190 ± 0.001 | 0.291 ± 0.003 | 0.347 ± 0.024 | - | - |
| large | `matmul` | f64 | 8 | `512x512` | - | 1.348 ± 0.010 | 1.375 ± 0.010 | 2.252 ± 0.024 | 1.815 ± 0.102 | - | - |
| large | `matmul_rect` | f64 | 8 | `1024x256 * 256x1024` | - | 2.734 ± 0.009 | 2.727 ± 0.009 | 3.495 ± 0.022 | 3.410 ± 0.119 | - | - |
| large | `matmul_rect` | f64 | 8 | `256x1024 * 1024x256` | - | 0.712 ± 0.003 | 0.715 ± 0.004 | 0.914 ± 0.007 | 1.093 ± 0.063 | - | - |
| large | `qr` | f64 | 8 | `1024x1024` | - | 69.209 ± 0.282 | 68.395 ± 0.153 | 49.873 ± 0.667 | 70.163 ± 0.721 | - | - |
| large | `qr` | f64 | 8 | `128x128` | - | 11.033 ± 0.010 | 0.785 ± 0.003 | 0.447 ± 0.002 | 0.884 ± 0.006 | - | - |
| large | `qr` | f64 | 8 | `256x256` | - | 13.519 ± 1.522 | 18.001 ± 0.034 | 1.869 ± 0.032 | 3.314 ± 0.136 | - | - |
| large | `qr` | f64 | 8 | `512x512` | - | 35.609 ± 9.774 | 27.006 ± 5.962 | 9.470 ± 0.065 | 16.457 ± 0.136 | - | - |
| large | `qr` | f64 | 8 | `64x64` | - | 0.125 ± 0.000 | 0.124 ± 0.000 | 0.160 ± 0.001 | 0.122 ± 0.002 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=1` | - | 7.309 ± 0.051 | 20.704 ± 7.378 | 97.909 ± 75.080 | 133.031 ± 16.468 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=16` | - | 8.436 ± 0.022 | 18.816 ± 0.021 | 117.206 ± 68.764 | 130.287 ± 17.379 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=64` | - | 8.984 ± 0.024 | 19.431 ± 32.803 | 111.074 ± 29.681 | 113.837 ± 9.549 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=1` | - | 0.201 ± 0.001 | 10.708 ± 0.009 | 0.179 ± 0.002 | 0.189 ± 0.004 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=16` | - | 0.217 ± 0.003 | 10.732 ± 0.008 | 8.809 ± 0.100 | 12.640 ± 0.743 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=64` | - | 0.245 ± 0.008 | 10.755 ± 0.008 | 8.815 ± 1.109 | 3.700 ± 0.236 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=1` | - | 0.478 ± 0.002 | 24.992 ± 13.327 | 21.951 ± 5.005 | 19.743 ± 0.264 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=16` | - | 0.519 ± 0.008 | 23.068 ± 6.245 | 34.642 ± 3.005 | 12.720 ± 6.966 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=64` | - | 0.580 ± 0.005 | 11.061 ± 8.983 | 30.652 ± 2.930 | 9.051 ± 66.131 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=1` | - | 1.699 ± 0.035 | 12.133 ± 2.999 | 64.870 ± 55.038 | 134.008 ± 136.787 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=16` | - | 1.928 ± 0.028 | 15.318 ± 0.992 | 60.127 ± 56.107 | 132.970 ± 21.944 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=64` | - | 2.119 ± 0.010 | 12.527 ± 0.053 | 43.991 ± 64.927 | 124.589 ± 122.722 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=1` | - | 0.041 ± 0.000 | 0.044 ± 0.000 | 0.053 ± 0.001 | 0.053 ± 0.001 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=16` | - | 0.046 ± 0.001 | 0.049 ± 0.000 | 0.066 ± 0.001 | 0.097 ± 0.002 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=64` | - | 0.053 ± 0.000 | 0.058 ± 0.000 | 0.078 ± 0.002 | 0.124 ± 0.009 | - | - |
| large | `svd` | f64 | 8 | `1024x1024` | - | 333.638 ± 6.180 | 327.535 ± 9.665 | 267.897 ± 4.667 | 384.489 ± 2.735 | - | - |
| large | `svd` | f64 | 8 | `128x128` | - | 13.529 ± 0.025 | 13.495 ± 0.018 | 2.814 ± 0.013 | 3.336 ± 0.019 | - | - |
| large | `svd` | f64 | 8 | `256x256` | - | 37.254 ± 4.081 | 36.844 ± 2.488 | 13.304 ± 0.045 | 15.523 ± 0.140 | - | - |
| large | `svd` | f64 | 8 | `512x512` | - | 83.691 ± 0.281 | 82.349 ± 2.207 | 54.689 ± 0.083 | 87.313 ± 0.927 | - | - |
| large | `svd` | f64 | 8 | `64x64` | - | 0.721 ± 0.002 | 0.717 ± 0.002 | 0.702 ± 0.004 | 0.765 ± 0.014 | - | - |
| small | `eigh` | f64 | 8 | `16x16` | - | 0.043 ± 0.000 | 0.042 ± 0.000 | 0.033 ± 0.000 | 0.046 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `2x2` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `32x32` | - | 0.127 ± 0.037 | 0.127 ± 0.000 | 0.113 ± 0.000 | 0.125 ± 0.001 | - | - |
| small | `eigh` | f64 | 8 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `8x8` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `16x16` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.149 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.146 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `32x32` | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.149 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.148 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.148 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | - | - | 0.069 ± 0.000 | 0.161 ± 0.005 | 0.042 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | - | - | 0.033 ± 0.000 | 0.126 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | - | - | 0.160 ± 0.000 | 0.263 ± 0.005 | 0.132 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | - | - | 0.036 ± 0.000 | 0.127 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | - | - | 0.043 ± 0.000 | 0.135 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | - | - | 0.068 ± 0.000 | 0.137 ± 0.001 | 0.043 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | - | - | 0.032 ± 0.000 | 0.108 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | - | - | 0.160 ± 0.000 | 0.229 ± 0.005 | 0.133 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | - | - | 0.034 ± 0.000 | 0.109 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | - | - | 0.041 ± 0.000 | 0.116 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | - | - | 0.047 ± 0.000 | 0.233 ± 0.004 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | - | - | 0.041 ± 0.000 | 0.214 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | - | - | 0.065 ± 0.000 | 0.350 ± 0.005 | 0.122 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | - | - | 0.041 ± 0.000 | 0.213 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | - | - | 0.043 ± 0.000 | 0.219 ± 0.005 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | - | - | 0.045 ± 0.000 | 0.184 ± 0.005 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | - | - | 0.037 ± 0.000 | 0.167 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | - | - | 0.066 ± 0.000 | 0.232 ± 0.007 | 0.071 ± 0.202 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | - | - | 0.038 ± 0.000 | 0.167 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | - | - | 0.040 ± 0.000 | 0.173 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.048 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.042 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.061 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.044 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.043 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | - | - | 0.069 ± 0.000 | 0.169 ± 0.007 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | - | - | 0.058 ± 0.000 | 0.146 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | - | - | 0.100 ± 0.000 | 0.218 ± 0.006 | 0.248 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | - | - | 0.060 ± 0.000 | 0.147 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | - | - | 0.062 ± 0.000 | 0.153 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | - | - | 0.069 ± 0.000 | 0.176 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | - | - | 0.057 ± 0.000 | 0.156 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | - | - | 0.102 ± 0.001 | 0.241 ± 0.006 | 0.069 ± 0.063 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | - | - | 0.059 ± 0.000 | 0.156 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | - | - | 0.062 ± 0.000 | 0.159 ± 0.006 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.070 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.065 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.082 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.067 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.068 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.342 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | - | - | 0.027 ± 0.000 | 0.328 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | - | - | 0.037 ± 0.000 | 0.354 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | - | - | 0.027 ± 0.000 | 0.328 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | - | - | 0.028 ± 0.000 | 0.333 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | - | - | 0.030 ± 0.000 | 0.151 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | - | - | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | - | - | 0.037 ± 0.000 | 0.169 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | - | - | 0.027 ± 0.000 | 0.142 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | - | - | 0.028 ± 0.000 | 0.145 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.000 | 0.108 ± 0.000 | 0.072 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.048 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.216 ± 0.001 | 0.271 ± 0.001 | 0.206 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.053 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.067 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | - | - | 0.082 ± 0.000 | 0.266 ± 0.005 | 0.069 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | - | - | 0.023 ± 0.000 | 0.185 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | - | - | 0.217 ± 0.000 | 0.456 ± 0.005 | 0.203 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | - | - | 0.027 ± 0.000 | 0.194 ± 0.007 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | - | - | 0.038 ± 0.000 | 0.213 ± 0.006 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | - | - | 0.080 ± 0.000 | 0.183 ± 0.006 | 0.069 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | - | - | 0.021 ± 0.000 | 0.117 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | - | - | 0.215 ± 0.001 | 0.356 ± 0.005 | 0.203 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | - | - | 0.025 ± 0.000 | 0.122 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | - | - | 0.036 ± 0.000 | 0.139 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `16x16` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `32x32` | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `64x64` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.016 ± 0.000 | 0.037 ± 0.003 | - | - |
| small | `matmul` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `16x16` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `qr` | f64 | 8 | `2x2` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `32x32` | - | 0.037 ± 0.000 | 0.035 ± 0.000 | 0.036 ± 0.000 | 0.038 ± 0.003 | - | - |
| small | `qr` | f64 | 8 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=4` | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=1` | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=4` | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.030 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=4` | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `16x16` | - | 0.068 ± 0.000 | 0.067 ± 0.000 | 0.065 ± 0.000 | 0.073 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `2x2` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `32x32` | - | 0.206 ± 0.001 | 0.205 ± 0.001 | 0.222 ± 0.001 | 0.204 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `4x4` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `8x8` | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 25.0x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch256 (native batch layout)`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 23.1x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-trace`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 7.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 1.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 76.238 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `tenferro-trace` is 42.5x faster than the slowest successful cell (`pytorch-cpu`, 76.238 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_eigh_jvp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`tenferro-trace`, 53.968 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_eigh_vjp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-trace`, 58.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=8, shape=`256x256`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 74.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 74.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 25.0x faster than the slowest successful cell (`tenferro-trace`, 53.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=8, shape=`256x256`): `jax-cpu` is 37.3x faster than the slowest successful cell (`tenferro-trace`, 13.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 51.9x faster than the slowest successful cell (`tenferro-trace`, 13.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=8, shape=`256x256`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 15.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=8, shape=`256x256`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-trace`, 95.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 27.9x faster than the slowest successful cell (`tenferro-trace`, 95.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=8, shape=`256x256`): `pytorch-cpu` is 18.8x faster than the slowest successful cell (`tenferro-trace`, 68.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`128x128,rhs=1`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-trace`, 10.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`128x128,rhs=1`): `pytorch-cpu` is 39.5x faster than the slowest successful cell (`tenferro-trace`, 10.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 29.816 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 134.502 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 34.7x faster than the slowest successful cell (`tenferro-trace`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=8, shape=`256x256,rhs=1`): `pytorch-cpu` is 25.3x faster than the slowest successful cell (`tenferro-trace`, 19.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 33.9x faster than the slowest successful cell (`tenferro-trace`, 20.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=8, shape=`256x256,rhs=1`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`tenferro-trace`, 20.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=8, shape=`128x128`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 11.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=8, shape=`128x128`): `pytorch-cpu` is 24.7x faster than the slowest successful cell (`tenferro-eager`, 11.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/qr` (f64, threads=8, shape=`128x128`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 11.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 133.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-eager` is 15.4x faster than the slowest successful cell (`jax-cpu`, 130.287 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 113.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=1`): `jax-cpu` is 56.6x faster than the slowest successful cell (`tenferro-trace`, 10.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=1`): `pytorch-cpu` is 59.9x faster than the slowest successful cell (`tenferro-trace`, 10.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=1`): `tenferro-eager` is 53.4x faster than the slowest successful cell (`tenferro-trace`, 10.708 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-eager` is 58.3x faster than the slowest successful cell (`jax-cpu`, 12.640 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-eager` is 43.9x faster than the slowest successful cell (`tenferro-trace`, 10.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-eager` is 52.3x faster than the slowest successful cell (`tenferro-trace`, 24.992 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-eager` is 66.8x faster than the slowest successful cell (`pytorch-cpu`, 34.642 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-eager` is 52.8x faster than the slowest successful cell (`pytorch-cpu`, 30.652 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-eager` is 78.9x faster than the slowest successful cell (`jax-cpu`, 134.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`jax-cpu`, 134.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-eager` is 69.0x faster than the slowest successful cell (`jax-cpu`, 132.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-eager` is 58.8x faster than the slowest successful cell (`jax-cpu`, 124.589 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-eager` is 25.0x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-eager` is 31.3x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-eager` is 29.3x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`32x32,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261031_085719/run.yaml`
- Timestamp: `2026-10-30T23:57:19.436239+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261031_085719`.

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

- tenferro-rs features: `blas-openblas`
- BLAS implementation: `openblas`
- BLAS version: `unknown`
- BLAS root: `/usr`
- BLAS library: `/usr`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261031_085719/cpu_ops_t1_20261031_085719.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 34.514 ± 0.044 | 34.369 ± 0.139 | 27.131 ± 0.065 | 34.072 ± 0.170 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.544 ± 0.001 | 0.542 ± 0.001 | 0.407 ± 0.001 | 0.551 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 8.561 ± 0.016 | 8.660 ± 0.011 | 6.810 ± 0.017 | 8.587 ± 0.079 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 2.139 ± 0.002 | 2.155 ± 0.015 | 1.685 ± 0.007 | 2.154 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.480 ± 0.001 | 0.494 ± 0.003 | 0.498 ± 0.001 | 0.429 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.127 ± 0.000 | 0.124 ± 0.000 | 0.131 ± 0.000 | 0.116 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.038 ± 0.000 | 0.036 ± 0.000 | 0.039 ± 0.000 | 0.046 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.727 ± 0.003 | 2.728 ± 0.007 | 2.631 ± 0.010 | 2.585 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.049 ± 0.000 | 0.049 ± 0.000 | 0.051 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.687 ± 0.002 | 0.685 ± 0.001 | 0.668 ± 0.008 | 0.660 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.177 ± 0.000 | 0.176 ± 0.000 | 0.171 ± 0.000 | 0.173 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 8.951 ± 0.016 | 9.014 ± 0.009 | 7.923 ± 0.028 | 8.818 ± 0.028 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.148 ± 0.000 | 0.147 ± 0.000 | 0.130 ± 0.000 | 0.159 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 2.242 ± 0.003 | 2.239 ± 0.004 | 1.981 ± 0.006 | 2.226 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.567 ± 0.001 | 0.567 ± 0.000 | 0.498 ± 0.001 | 0.569 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.321 ± 0.004 | 0.333 ± 0.004 | 1.120 ± 0.008 | 1.086 ± 0.029 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.031 ± 0.001 | 0.176 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.085 ± 0.001 | 0.085 ± 0.001 | 0.280 ± 0.006 | 0.359 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.025 ± 0.000 | 0.024 ± 0.000 | 0.078 ± 0.003 | 0.209 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.040 ± 0.001 | 0.040 ± 0.000 | 0.043 ± 0.001 | 0.174 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.147 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.168 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.166 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.051 ± 0.000 | 0.049 ± 0.000 | 0.163 ± 0.001 | 0.185 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.167 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.052 ± 0.000 | 0.174 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.024 ± 0.000 | 0.167 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.083 ± 0.001 | 0.079 ± 0.001 | 0.252 ± 0.002 | 0.304 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.169 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.073 ± 0.002 | 0.192 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.030 ± 0.000 | 0.172 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 7.971 ± 0.015 | 7.971 ± 0.009 | 5.836 ± 0.054 | 8.325 ± 0.191 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.131 ± 0.001 | 0.135 ± 0.000 | 0.097 ± 0.000 | 0.143 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 1.997 ± 0.003 | 2.010 ± 0.010 | 1.485 ± 0.011 | 2.073 ± 0.029 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.505 ± 0.001 | 0.502 ± 0.009 | 0.374 ± 0.003 | 0.527 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.370 ± 0.001 | 0.366 ± 0.001 | 0.244 ± 0.001 | 0.338 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.099 ± 0.000 | 0.096 ± 0.000 | 0.066 ± 0.000 | 0.094 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.031 ± 0.000 | 0.029 ± 0.000 | 0.021 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.005 ± 0.001 | 0.996 ± 0.002 | 0.668 ± 0.001 | 0.954 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.256 ± 0.000 | 0.256 ± 0.000 | 0.172 ± 0.001 | 0.247 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.070 ± 0.000 | 0.069 ± 0.000 | 0.047 ± 0.000 | 0.076 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 2.653 ± 0.007 | 2.652 ± 0.012 | 1.733 ± 0.005 | 2.714 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.049 ± 0.000 | 0.048 ± 0.000 | 0.033 ± 0.000 | 0.060 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.670 ± 0.002 | 0.669 ± 0.001 | 0.438 ± 0.002 | 0.695 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.177 ± 0.000 | 0.172 ± 0.000 | 0.114 ± 0.000 | 0.182 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.652 ± 0.011 | 2.695 ± 0.012 | 1.878 ± 0.021 | 2.922 ± 0.021 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.051 ± 0.000 | 0.050 ± 0.000 | 0.049 ± 0.001 | 0.068 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.668 ± 0.001 | 0.675 ± 0.004 | 0.496 ± 0.008 | 0.728 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.173 ± 0.000 | 0.173 ± 0.000 | 0.137 ± 0.002 | 0.194 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.307 ± 0.000 | 0.273 ± 0.001 | 0.106 ± 0.001 | 0.343 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.084 ± 0.000 | 0.075 ± 0.000 | 0.041 ± 0.000 | 0.096 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.040 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.482 ± 0.001 | 0.442 ± 0.001 | 0.156 ± 0.002 | 0.487 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.128 ± 0.000 | 0.117 ± 0.000 | 0.053 ± 0.001 | 0.133 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.039 ± 0.000 | 0.036 ± 0.000 | 0.028 ± 0.000 | 0.050 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.982 ± 0.001 | 0.958 ± 0.001 | 0.548 ± 0.005 | 1.014 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.251 ± 0.000 | 0.240 ± 0.001 | 0.151 ± 0.002 | 0.261 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.070 ± 0.000 | 0.067 ± 0.000 | 0.052 ± 0.001 | 0.082 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 58.644 ± 0.112 | 59.869 ± 0.213 | 59.847 ± 0.104 | 59.482 ± 0.280 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.924 ± 0.003 | 0.932 ± 0.002 | 0.918 ± 0.002 | 0.957 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 14.638 ± 0.031 | 14.674 ± 0.031 | 14.991 ± 0.025 | 14.925 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 3.665 ± 0.009 | 3.654 ± 0.010 | 3.744 ± 0.011 | 3.755 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.675 ± 0.018 | 1.682 ± 0.008 | 1.410 ± 0.009 | 1.544 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.032 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.426 ± 0.001 | 0.422 ± 0.002 | 0.360 ± 0.001 | 0.400 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.113 ± 0.000 | 0.113 ± 0.000 | 0.098 ± 0.000 | 0.108 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 5.071 ± 0.013 | 5.044 ± 0.009 | 5.355 ± 0.025 | 4.993 ± 0.025 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.089 ± 0.000 | 0.087 ± 0.000 | 0.093 ± 0.000 | 0.092 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 1.276 ± 0.003 | 1.269 ± 0.001 | 1.346 ± 0.007 | 1.277 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.324 ± 0.001 | 0.322 ± 0.000 | 0.342 ± 0.000 | 0.336 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 15.839 ± 0.035 | 15.846 ± 0.042 | 18.250 ± 0.038 | 15.952 ± 0.033 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.257 ± 0.001 | 0.255 ± 0.001 | 0.292 ± 0.001 | 0.267 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 3.965 ± 0.006 | 3.961 ± 0.008 | 4.563 ± 0.008 | 4.020 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.997 ± 0.002 | 0.996 ± 0.002 | 1.152 ± 0.007 | 1.020 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.833 ± 0.018 | 10.315 ± 0.033 | 2.990 ± 0.022 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.001 | 0.266 ± 0.002 | 0.065 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.454 ± 0.003 | 2.713 ± 0.031 | 0.722 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.120 ± 0.000 | 0.748 ± 0.005 | 0.185 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.251 ± 0.000 | 0.192 ± 0.001 | 0.046 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.106 ± 0.000 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.000 | 0.126 ± 0.001 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.109 ± 0.001 | 0.025 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.325 ± 0.000 | 0.564 ± 0.005 | 0.560 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.111 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.092 ± 0.000 | 0.219 ± 0.003 | 0.153 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.132 ± 0.001 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.487 ± 0.002 | 7.213 ± 0.037 | 0.977 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.220 ± 0.001 | 0.035 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.131 ± 0.001 | 1.973 ± 0.007 | 0.253 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.556 ± 0.002 | 0.082 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.423 ± 0.011 | 4.148 ± 0.152 | 4.838 ± 0.075 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.095 ± 0.000 | 0.138 ± 0.001 | 0.107 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.123 ± 0.001 | 1.047 ± 0.024 | 1.225 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.296 ± 0.001 | 0.321 ± 0.003 | 0.336 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.602 ± 0.001 | 0.231 ± 0.005 | 0.701 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.076 ± 0.000 | 0.028 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.171 ± 0.000 | 0.115 ± 0.001 | 0.190 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.064 ± 0.000 | 0.087 ± 0.001 | 0.068 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.860 ± 0.001 | 0.376 ± 0.007 | 0.990 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.081 ± 0.001 | 0.042 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.235 ± 0.000 | 0.149 ± 0.001 | 0.264 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.080 ± 0.000 | 0.095 ± 0.001 | 0.090 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.706 ± 0.003 | 1.314 ± 0.011 | 1.844 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.000 | 0.096 ± 0.001 | 0.057 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.443 ± 0.001 | 0.393 ± 0.004 | 0.483 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.131 ± 0.000 | 0.154 ± 0.002 | 0.141 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 300.945 ± 0.642 | 300.109 ± 1.037 | 382.388 ± 2.986 | 319.964 ± 6.346 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 1.772 ± 0.004 | 1.772 ± 0.004 | 1.858 ± 0.007 | 1.928 ± 0.016 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 8.820 ± 0.019 | 8.855 ± 0.037 | 9.995 ± 0.099 | 9.401 ± 0.043 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 50.816 ± 0.117 | 50.408 ± 0.159 | 59.144 ± 0.263 | 53.622 ± 0.112 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.435 ± 0.001 | 0.434 ± 0.001 | 0.401 ± 0.002 | 0.470 ± 0.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 389.396 ± 0.794 | 624.841 ± 2.643 | 480.201 ± 1.613 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 10.475 ± 0.015 | 14.023 ± 0.056 | 11.755 ± 0.045 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 63.148 ± 0.078 | 93.677 ± 0.256 | 75.632 ± 0.517 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 391.148 ± 1.035 | 450.112 ± 0.793 | 480.646 ± 1.241 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 10.532 ± 0.021 | 11.331 ± 0.020 | 11.738 ± 0.016 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 63.194 ± 0.188 | 72.612 ± 0.116 | 74.227 ± 0.218 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 375.255 ± 0.358 | 752.922 ± 1.399 | 353.788 ± 1.223 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 7.149 ± 0.006 | 13.835 ± 0.052 | 6.798 ± 0.022 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 50.740 ± 0.083 | 107.067 ± 0.150 | 47.387 ± 0.061 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 378.214 ± 0.327 | 399.549 ± 0.808 | 349.546 ± 1.070 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 7.315 ± 0.026 | 7.426 ± 0.044 | 6.448 ± 0.059 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 51.424 ± 0.064 | 56.029 ± 0.664 | 46.578 ± 0.083 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 81.186 ± 0.044 | 79.672 ± 0.146 | 79.708 ± 0.153 | 77.365 ± 0.791 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.209 ± 0.001 | 0.183 ± 0.000 | 0.198 ± 0.004 | 0.207 ± 0.003 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 1.400 ± 0.002 | 1.327 ± 0.003 | 1.328 ± 0.017 | 1.395 ± 0.009 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 10.498 ± 0.100 | 10.207 ± 0.058 | 11.216 ± 0.033 | 10.465 ± 0.030 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.037 ± 0.000 | 0.024 ± 0.000 | 0.035 ± 0.001 | 0.062 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 241.094 ± 0.356 | 236.731 ± 0.401 | 229.978 ± 0.428 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.564 ± 0.002 | 0.611 ± 0.009 | 0.529 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.042 ± 0.027 | 3.921 ± 0.022 | 3.892 ± 0.013 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 30.930 ± 0.037 | 32.135 ± 0.066 | 29.677 ± 0.085 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.077 ± 0.000 | 0.129 ± 0.002 | 0.095 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 465.428 ± 0.871 | 518.404 ± 1.995 | 497.811 ± 1.566 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 9.397 ± 0.032 | 10.501 ± 0.036 | 9.276 ± 0.036 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 66.534 ± 0.083 | 79.651 ± 0.300 | 67.254 ± 0.107 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 466.968 ± 0.863 | 505.869 ± 0.819 | 497.327 ± 4.005 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 9.485 ± 0.020 | 10.551 ± 0.036 | 9.207 ± 0.036 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 66.540 ± 0.286 | 83.520 ± 0.091 | 68.021 ± 0.198 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 36.732 ± 0.063 | 48.335 ± 0.202 | 40.877 ± 0.080 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.206 ± 0.001 | 0.265 ± 0.002 | 0.224 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.947 ± 0.005 | 1.185 ± 0.013 | 1.041 ± 0.006 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 5.489 ± 0.012 | 7.818 ± 0.038 | 6.964 ± 0.051 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.070 ± 0.000 | 0.116 ± 0.003 | 0.079 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 35.734 ± 0.111 | 48.410 ± 0.175 | 41.192 ± 0.070 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.881 ± 0.003 | 1.388 ± 0.014 | 0.950 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 5.269 ± 0.019 | 7.551 ± 0.030 | 6.569 ± 0.026 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 35.699 ± 0.083 | 84.417 ± 0.224 | 40.316 ± 0.057 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.886 ± 0.007 | 2.039 ± 0.015 | 0.967 ± 0.002 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 5.330 ± 0.084 | 12.948 ± 0.076 | 6.610 ± 0.025 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 874.438 ± 15.396 | 958.752 ± 3.611 | 901.207 ± 1.653 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.518 ± 0.024 | 4.064 ± 0.019 | 3.589 ± 0.018 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 20.380 ± 0.028 | 21.654 ± 0.215 | 20.656 ± 0.070 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 139.198 ± 0.304 | 161.921 ± 1.573 | 144.522 ± 0.750 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.736 ± 0.005 | 0.887 ± 0.008 | 0.805 ± 0.009 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 862.054 ± 9.461 | 1137.682 ± 3.463 | 895.931 ± 1.944 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 20.391 ± 0.039 | 25.723 ± 0.068 | 20.662 ± 0.035 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 140.448 ± 0.201 | 189.718 ± 0.452 | 142.662 ± 0.284 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 862.034 ± 0.573 | 869.346 ± 5.966 | 913.789 ± 3.374 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 20.370 ± 0.034 | 21.500 ± 0.056 | 20.631 ± 0.048 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 140.538 ± 0.319 | 161.918 ± 1.178 | 143.724 ± 0.228 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 79.746 ± 0.148 | 79.893 ± 0.138 | 78.025 ± 3.547 | 76.376 ± 0.300 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.182 ± 0.001 | 0.181 ± 0.000 | 0.239 ± 0.002 | 0.207 ± 0.005 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 1.310 ± 0.002 | 1.315 ± 0.004 | 1.365 ± 0.024 | 1.500 ± 0.019 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 10.140 ± 0.022 | 10.154 ± 0.020 | 10.539 ± 0.035 | 9.967 ± 0.034 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 19.988 ± 0.035 | 20.041 ± 0.046 | 19.475 ± 0.243 | 19.212 ± 0.104 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 5.191 ± 0.004 | 5.224 ± 0.012 | 5.297 ± 0.020 | 5.224 ± 0.028 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 159.953 ± 0.194 | 160.945 ± 0.249 | 202.513 ± 3.905 | 175.537 ± 0.358 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.731 ± 0.001 | 0.729 ± 0.001 | 0.829 ± 0.006 | 0.769 ± 0.002 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 3.715 ± 0.008 | 3.706 ± 0.011 | 4.933 ± 0.091 | 4.022 ± 0.059 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 25.211 ± 0.064 | 25.424 ± 0.035 | 33.690 ± 0.084 | 28.051 ± 0.080 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.122 ± 0.000 | 0.121 ± 0.000 | 0.136 ± 0.001 | 0.126 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 33.893 ± 0.063 | 34.252 ± 0.031 | 46.735 ± 0.149 | 40.318 ± 0.096 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 35.637 ± 0.181 | 36.056 ± 0.061 | 48.675 ± 0.204 | 41.795 ± 0.073 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 40.425 ± 0.106 | 40.870 ± 0.089 | 53.545 ± 0.212 | 46.703 ± 0.119 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.151 ± 0.000 | 0.152 ± 0.000 | 0.178 ± 0.005 | 0.188 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.202 ± 0.001 | 0.207 ± 0.001 | 0.234 ± 0.005 | 0.280 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.357 ± 0.005 | 0.364 ± 0.005 | 0.381 ± 0.004 | 0.455 ± 0.007 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.789 ± 0.003 | 0.818 ± 0.002 | 1.017 ± 0.021 | 0.957 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.955 ± 0.004 | 0.994 ± 0.007 | 1.170 ± 0.015 | 1.145 ± 0.022 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 1.430 ± 0.004 | 1.470 ± 0.006 | 1.653 ± 0.023 | 1.660 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 4.946 ± 0.013 | 5.004 ± 0.045 | 6.718 ± 0.179 | 6.761 ± 0.066 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 5.489 ± 0.036 | 5.595 ± 0.072 | 7.243 ± 0.034 | 7.214 ± 0.032 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 6.933 ± 0.023 | 7.015 ± 0.067 | 8.774 ± 0.024 | 8.778 ± 0.075 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.039 ± 0.000 | 0.041 ± 0.000 | 0.053 ± 0.001 | 0.059 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.057 ± 0.001 | 0.060 ± 0.001 | 0.070 ± 0.001 | 0.092 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.114 ± 0.000 | 0.118 ± 0.000 | 0.119 ± 0.001 | 0.176 ± 0.007 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 697.946 ± 7.964 | 707.708 ± 16.150 | 792.583 ± 1.212 | 720.432 ± 1.676 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 3.164 ± 0.008 | 3.144 ± 0.008 | 3.578 ± 0.017 | 3.330 ± 0.012 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 17.855 ± 0.066 | 17.778 ± 0.113 | 19.455 ± 0.117 | 18.329 ± 0.030 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 119.077 ± 0.398 | 118.806 ± 0.280 | 125.582 ± 0.240 | 122.111 ± 0.345 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.684 ± 0.002 | 0.683 ± 0.001 | 0.764 ± 0.010 | 0.752 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.042 ± 0.000 | 0.041 ± 0.000 | 0.033 ± 0.000 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.127 ± 0.000 | 0.125 ± 0.000 | 0.106 ± 0.000 | 0.134 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.167 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.144 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.007 ± 0.001 | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.172 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.146 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.164 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.061 ± 0.000 | 0.161 ± 0.005 | 0.053 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.026 ± 0.000 | 0.123 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.151 ± 0.000 | 0.253 ± 0.004 | 0.142 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.028 ± 0.000 | 0.125 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.035 ± 0.000 | 0.134 ± 0.002 | 0.025 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.061 ± 0.000 | 0.135 ± 0.001 | 0.053 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.025 ± 0.000 | 0.106 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.151 ± 0.000 | 0.221 ± 0.005 | 0.142 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.027 ± 0.000 | 0.107 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.034 ± 0.000 | 0.113 ± 0.001 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.037 ± 0.000 | 0.223 ± 0.006 | 0.026 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.000 | 0.202 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.059 ± 0.000 | 0.276 ± 0.006 | 0.051 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.031 ± 0.000 | 0.201 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.000 | 0.204 ± 0.004 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.037 ± 0.000 | 0.174 ± 0.005 | 0.027 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.029 ± 0.000 | 0.159 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.060 ± 0.000 | 0.214 ± 0.005 | 0.052 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.029 ± 0.000 | 0.160 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.031 ± 0.000 | 0.161 ± 0.006 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.047 ± 0.000 | 0.024 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.042 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.059 ± 0.000 | 0.037 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.044 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.044 ± 0.000 | 0.025 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.056 ± 0.000 | 0.161 ± 0.006 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.045 ± 0.000 | 0.139 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.092 ± 0.000 | 0.203 ± 0.005 | 0.062 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.046 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.048 ± 0.000 | 0.142 ± 0.002 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.056 ± 0.000 | 0.164 ± 0.004 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.044 ± 0.000 | 0.147 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.093 ± 0.000 | 0.206 ± 0.004 | 0.062 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.045 ± 0.000 | 0.148 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.048 ± 0.000 | 0.150 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.031 ± 0.000 | 0.071 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.066 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.084 ± 0.001 | 0.038 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.068 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.068 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.024 ± 0.000 | 0.340 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.021 ± 0.000 | 0.327 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.031 ± 0.008 | 0.358 ± 0.004 | 0.029 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.021 ± 0.000 | 0.327 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.022 ± 0.000 | 0.329 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.024 ± 0.000 | 0.149 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.020 ± 0.000 | 0.142 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.031 ± 0.000 | 0.170 ± 0.004 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.021 ± 0.000 | 0.141 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.021 ± 0.000 | 0.144 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.000 | 0.109 ± 0.000 | 0.081 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.049 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.210 ± 0.000 | 0.260 ± 0.002 | 0.214 ± 0.006 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.054 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.068 ± 0.001 | 0.036 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.078 ± 0.000 | 0.266 ± 0.003 | 0.078 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.019 ± 0.000 | 0.185 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.211 ± 0.001 | 0.442 ± 0.007 | 0.209 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.022 ± 0.000 | 0.194 ± 0.007 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.034 ± 0.000 | 0.212 ± 0.006 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.076 ± 0.000 | 0.182 ± 0.005 | 0.078 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.000 | 0.118 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.211 ± 0.001 | 0.344 ± 0.008 | 0.211 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.021 ± 0.000 | 0.121 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.032 ± 0.000 | 0.136 ± 0.001 | 0.034 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.039 ± 0.002 | 0.049 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.034 ± 0.000 | 0.034 ± 0.000 | 0.031 ± 0.000 | 0.046 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.028 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.001 | 0.033 ± 0.008 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.066 ± 0.000 | 0.065 ± 0.000 | 0.066 ± 0.000 | 0.073 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.204 ± 0.001 | 0.203 ± 0.000 | 0.209 ± 0.000 | 0.213 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 25.3x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 33.7x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 22.6x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 29.5x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 7.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 1.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.556 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 36.0x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 35.9x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 25.5x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 35.7x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 37.2x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.0x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 22.2x faster than the slowest successful cell (`pytorch-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.329 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

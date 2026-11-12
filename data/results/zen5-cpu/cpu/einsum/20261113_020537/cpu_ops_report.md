# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_020537/cpu_ops_t1_20261113_020537.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 34.144 ± 0.163 | 33.396 ± 0.057 | 27.410 ± 0.034 | 34.183 ± 0.351 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.512 ± 0.001 | 0.511 ± 0.003 | 0.412 ± 0.001 | 0.550 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 8.556 ± 0.016 | 8.350 ± 0.013 | 6.880 ± 0.022 | 8.587 ± 0.103 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 2.102 ± 0.007 | 2.063 ± 0.003 | 1.700 ± 0.004 | 2.162 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.521 ± 0.001 | 0.518 ± 0.001 | 0.499 ± 0.002 | 0.429 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.135 ± 0.000 | 0.133 ± 0.000 | 0.131 ± 0.001 | 0.117 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.037 ± 0.000 | 0.039 ± 0.000 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.855 ± 0.005 | 2.837 ± 0.008 | 2.666 ± 0.002 | 2.579 ± 0.010 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.048 ± 0.000 | 0.049 ± 0.000 | 0.052 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.727 ± 0.002 | 0.714 ± 0.001 | 0.673 ± 0.001 | 0.664 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.185 ± 0.000 | 0.183 ± 0.000 | 0.174 ± 0.001 | 0.174 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 8.832 ± 0.010 | 8.823 ± 0.021 | 7.970 ± 0.021 | 8.843 ± 0.033 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.141 ± 0.001 | 0.139 ± 0.000 | 0.131 ± 0.000 | 0.153 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 2.210 ± 0.003 | 2.198 ± 0.008 | 1.994 ± 0.003 | 2.230 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.556 ± 0.001 | 0.550 ± 0.001 | 0.503 ± 0.001 | 0.570 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.443 ± 0.005 | 0.458 ± 0.003 | 1.131 ± 0.031 | 1.031 ± 0.033 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.032 ± 0.001 | 0.179 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.114 ± 0.001 | 0.114 ± 0.002 | 0.278 ± 0.004 | 0.364 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.032 ± 0.000 | 0.031 ± 0.000 | 0.080 ± 0.002 | 0.214 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.145 ± 0.000 | 0.148 ± 0.000 | 0.044 ± 0.001 | 0.179 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.148 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.022 ± 0.000 | 0.171 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.167 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.158 ± 0.001 | 0.157 ± 0.001 | 0.164 ± 0.003 | 0.197 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.168 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.044 ± 0.001 | 0.042 ± 0.000 | 0.052 ± 0.001 | 0.176 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.001 | 0.171 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.180 ± 0.001 | 0.178 ± 0.001 | 0.251 ± 0.008 | 0.305 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.000 | 0.171 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.048 ± 0.000 | 0.048 ± 0.000 | 0.072 ± 0.001 | 0.203 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.001 | 0.178 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 9.517 ± 0.014 | 9.636 ± 0.018 | 5.879 ± 0.039 | 8.329 ± 0.185 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.156 ± 0.000 | 0.155 ± 0.000 | 0.097 ± 0.000 | 0.143 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.399 ± 0.006 | 2.395 ± 0.007 | 1.479 ± 0.019 | 2.076 ± 0.041 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.606 ± 0.001 | 0.606 ± 0.002 | 0.374 ± 0.003 | 0.533 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.315 ± 0.001 | 0.314 ± 0.000 | 0.244 ± 0.001 | 0.340 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.084 ± 0.000 | 0.082 ± 0.000 | 0.066 ± 0.000 | 0.095 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.024 ± 0.000 | 0.021 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.808 ± 0.002 | 0.800 ± 0.001 | 0.668 ± 0.002 | 0.957 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.207 ± 0.001 | 0.204 ± 0.000 | 0.172 ± 0.001 | 0.247 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.055 ± 0.000 | 0.047 ± 0.001 | 0.076 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 2.535 ± 0.010 | 2.535 ± 0.006 | 1.728 ± 0.005 | 2.718 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.046 ± 0.000 | 0.044 ± 0.000 | 0.033 ± 0.001 | 0.060 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.643 ± 0.001 | 0.650 ± 0.004 | 0.435 ± 0.001 | 0.695 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.165 ± 0.000 | 0.164 ± 0.001 | 0.113 ± 0.000 | 0.182 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.202 ± 0.012 | 2.234 ± 0.008 | 1.898 ± 0.022 | 2.872 ± 0.040 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.043 ± 0.000 | 0.043 ± 0.000 | 0.049 ± 0.001 | 0.068 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.553 ± 0.003 | 0.562 ± 0.004 | 0.497 ± 0.008 | 0.724 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.147 ± 0.001 | 0.144 ± 0.000 | 0.137 ± 0.002 | 0.193 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.210 ± 0.000 | 0.160 ± 0.000 | 0.107 ± 0.001 | 0.342 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.059 ± 0.000 | 0.047 ± 0.000 | 0.042 ± 0.001 | 0.096 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.041 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.323 ± 0.001 | 0.271 ± 0.000 | 0.155 ± 0.002 | 0.488 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.087 ± 0.000 | 0.075 ± 0.000 | 0.053 ± 0.001 | 0.134 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.001 | 0.049 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.839 ± 0.002 | 0.800 ± 0.001 | 0.553 ± 0.007 | 1.015 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.000 | 0.035 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.213 ± 0.001 | 0.203 ± 0.001 | 0.150 ± 0.001 | 0.261 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.000 | 0.057 ± 0.000 | 0.052 ± 0.001 | 0.082 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 57.596 ± 0.060 | 57.844 ± 0.045 | 59.993 ± 0.046 | 59.429 ± 0.429 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.899 ± 0.004 | 0.898 ± 0.001 | 0.920 ± 0.004 | 0.959 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 14.458 ± 0.017 | 14.457 ± 0.019 | 15.019 ± 0.064 | 14.961 ± 0.046 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 3.647 ± 0.005 | 3.609 ± 0.007 | 3.742 ± 0.013 | 3.752 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.301 ± 0.004 | 1.264 ± 0.003 | 1.434 ± 0.008 | 1.540 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.032 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.324 ± 0.001 | 0.321 ± 0.001 | 0.366 ± 0.001 | 0.402 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.086 ± 0.000 | 0.085 ± 0.000 | 0.099 ± 0.000 | 0.109 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 4.603 ± 0.015 | 4.597 ± 0.010 | 5.470 ± 0.014 | 5.006 ± 0.023 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.078 ± 0.000 | 0.076 ± 0.000 | 0.094 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 1.155 ± 0.001 | 1.163 ± 0.003 | 1.378 ± 0.004 | 1.276 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.295 ± 0.001 | 0.293 ± 0.001 | 0.350 ± 0.001 | 0.336 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 15.024 ± 0.023 | 15.112 ± 0.032 | 18.391 ± 0.037 | 15.980 ± 0.083 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.239 ± 0.000 | 0.243 ± 0.001 | 0.293 ± 0.001 | 0.267 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 3.805 ± 0.008 | 3.805 ± 0.010 | 4.600 ± 0.009 | 4.026 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.951 ± 0.001 | 0.947 ± 0.001 | 1.158 ± 0.004 | 1.023 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.675 ± 0.012 | 10.771 ± 0.059 | 2.956 ± 0.024 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.266 ± 0.003 | 0.064 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.424 ± 0.001 | 2.656 ± 0.018 | 0.731 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.114 ± 0.000 | 0.748 ± 0.006 | 0.186 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.477 ± 0.001 | 0.195 ± 0.001 | 0.045 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.107 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.130 ± 0.000 | 0.126 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.110 ± 0.001 | 0.025 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.501 ± 0.001 | 0.564 ± 0.007 | 0.558 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.111 ± 0.000 | 0.024 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.135 ± 0.000 | 0.220 ± 0.002 | 0.154 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.132 ± 0.001 | 0.055 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.613 ± 0.002 | 7.131 ± 0.058 | 0.975 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.219 ± 0.002 | 0.034 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.163 ± 0.001 | 1.922 ± 0.019 | 0.253 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.000 | 0.554 ± 0.004 | 0.083 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 8.545 ± 0.076 | 4.055 ± 0.274 | 4.804 ± 0.043 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.172 ± 0.000 | 0.136 ± 0.001 | 0.107 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.131 ± 0.008 | 1.062 ± 0.025 | 1.216 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.561 ± 0.001 | 0.323 ± 0.004 | 0.334 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.765 ± 0.002 | 0.233 ± 0.005 | 0.698 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.000 | 0.075 ± 0.001 | 0.028 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.224 ± 0.001 | 0.113 ± 0.001 | 0.190 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.088 ± 0.000 | 0.084 ± 0.000 | 0.067 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.144 ± 0.003 | 0.373 ± 0.007 | 0.981 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.060 ± 0.000 | 0.079 ± 0.001 | 0.043 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.320 ± 0.001 | 0.148 ± 0.002 | 0.262 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.112 ± 0.000 | 0.093 ± 0.001 | 0.088 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.151 ± 0.004 | 1.317 ± 0.015 | 1.832 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.091 ± 0.000 | 0.094 ± 0.001 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.810 ± 0.002 | 0.391 ± 0.003 | 0.482 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.230 ± 0.001 | 0.151 ± 0.001 | 0.140 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 281.937 ± 1.634 | 282.318 ± 0.403 | 383.516 ± 1.797 | 331.240 ± 5.521 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 2.203 ± 0.002 | 2.191 ± 0.004 | 1.859 ± 0.004 | 1.923 ± 0.013 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 9.426 ± 0.018 | 9.465 ± 0.020 | 9.858 ± 0.028 | 9.357 ± 0.074 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 47.714 ± 0.071 | 47.830 ± 0.068 | 59.123 ± 0.118 | 54.126 ± 0.137 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.446 ± 0.000 | 0.449 ± 0.001 | 0.402 ± 0.002 | 0.467 ± 0.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | 713.047 ± 0.956 | 496.973 ± 2.580 | 634.739 ± 3.511 | 491.925 ± 1.138 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 21.994 ± 0.075 | 16.280 ± 0.026 | 14.061 ± 0.037 | 11.671 ± 0.022 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 115.946 ± 0.280 | 83.762 ± 0.218 | 94.204 ± 0.425 | 74.578 ± 1.032 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | 428.316 ± 0.786 | 498.077 ± 2.209 | 459.244 ± 3.721 | 483.056 ± 2.886 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 12.140 ± 0.065 | 16.323 ± 0.033 | 11.388 ± 0.042 | 11.713 ± 0.043 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 67.665 ± 0.250 | 83.711 ± 0.202 | 72.911 ± 0.365 | 73.564 ± 0.205 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | 738.543 ± 1.107 | 351.497 ± 0.456 | 778.889 ± 1.588 | 354.277 ± 9.896 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 13.583 ± 0.081 | 6.552 ± 0.023 | 13.914 ± 0.044 | 6.825 ± 0.181 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 93.044 ± 0.168 | 46.259 ± 0.149 | 113.892 ± 0.241 | 48.460 ± 0.163 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | 725.783 ± 1.871 | 360.663 ± 1.507 | 407.188 ± 0.960 | 348.968 ± 0.821 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 12.230 ± 0.073 | 6.925 ± 0.022 | 7.460 ± 0.105 | 6.452 ± 0.011 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 90.845 ± 0.078 | 49.441 ± 0.089 | 60.853 ± 0.543 | 46.881 ± 0.086 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 71.170 ± 0.138 | 69.689 ± 0.088 | 86.620 ± 0.661 | 77.738 ± 0.663 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.166 ± 0.001 | 0.143 ± 0.000 | 0.199 ± 0.004 | 0.212 ± 0.003 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 1.153 ± 0.004 | 1.088 ± 0.002 | 1.331 ± 0.018 | 1.379 ± 0.012 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 8.802 ± 0.020 | 8.581 ± 0.008 | 10.840 ± 0.045 | 10.339 ± 0.034 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.035 ± 0.000 | 0.024 ± 0.000 | 0.035 ± 0.003 | 0.063 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 210.301 ± 0.972 | 251.836 ± 0.960 | 230.409 ± 0.786 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.437 ± 0.000 | 0.612 ± 0.009 | 0.532 ± 0.011 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.321 ± 0.005 | 3.904 ± 0.011 | 3.807 ± 0.045 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 26.074 ± 0.029 | 31.949 ± 0.134 | 29.737 ± 0.196 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.000 | 0.127 ± 0.001 | 0.094 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | 993.358 ± 7.614 | 543.030 ± 1.127 | 530.823 ± 2.216 | 495.998 ± 4.838 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 19.070 ± 0.044 | 10.516 ± 0.026 | 10.527 ± 0.024 | 9.268 ± 0.019 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 129.292 ± 0.254 | 73.858 ± 0.245 | 82.346 ± 0.412 | 67.401 ± 0.110 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | 857.598 ± 2.753 | 541.347 ± 0.861 | 514.695 ± 1.740 | 498.209 ± 5.790 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 14.528 ± 0.110 | 10.535 ± 0.028 | 10.620 ± 0.049 | 9.174 ± 0.015 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 108.075 ± 0.230 | 73.386 ± 0.094 | 86.139 ± 0.428 | 68.285 ± 0.276 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 84.773 ± 0.381 | 49.374 ± 0.149 | 40.751 ± 0.152 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.394 ± 0.001 | 0.262 ± 0.005 | 0.228 ± 0.007 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.953 ± 0.012 | 1.194 ± 0.026 | 1.043 ± 0.019 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 11.895 ± 0.029 | 7.249 ± 0.044 | 6.981 ± 0.046 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.124 ± 0.001 | 0.113 ± 0.002 | 0.080 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | 58.003 ± 0.134 | 56.732 ± 0.218 | 49.388 ± 0.242 | 41.448 ± 0.085 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.577 ± 0.035 | 1.294 ± 0.002 | 1.409 ± 0.032 | 0.977 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 8.935 ± 0.109 | 7.980 ± 0.017 | 7.453 ± 0.245 | 6.568 ± 0.015 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | 30.034 ± 0.146 | 56.136 ± 0.104 | 85.667 ± 4.050 | 40.276 ± 0.098 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.837 ± 0.031 | 1.296 ± 0.013 | 2.055 ± 0.038 | 0.969 ± 0.007 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.628 ± 0.130 | 7.913 ± 0.009 | 13.234 ± 0.123 | 6.612 ± 0.022 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1119.724 ± 2.324 | 958.051 ± 4.723 | 909.772 ± 8.103 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 5.027 ± 0.030 | 4.032 ± 0.012 | 3.646 ± 0.018 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 28.621 ± 0.033 | 21.499 ± 0.092 | 20.718 ± 0.099 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 167.503 ± 0.547 | 151.900 ± 1.616 | 142.932 ± 0.586 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.227 ± 0.001 | 0.883 ± 0.012 | 0.804 ± 0.004 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | 1537.732 ± 14.203 | 1096.173 ± 3.909 | 1154.158 ± 4.239 | 904.484 ± 1.715 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 39.084 ± 0.144 | 28.660 ± 0.040 | 25.814 ± 0.114 | 20.602 ± 0.106 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 228.792 ± 1.036 | 168.364 ± 0.594 | 196.733 ± 2.738 | 143.114 ± 0.377 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | 831.474 ± 16.232 | 1119.245 ± 1.448 | 870.117 ± 2.726 | 910.452 ± 2.690 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 20.406 ± 0.039 | 28.646 ± 0.060 | 21.552 ± 0.075 | 20.587 ± 0.040 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 122.716 ± 0.310 | 168.106 ± 0.442 | 157.251 ± 0.469 | 142.319 ± 0.290 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 69.675 ± 0.081 | 69.250 ± 0.141 | 79.076 ± 8.336 | 75.928 ± 0.378 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.141 ± 0.000 | 0.140 ± 0.000 | 0.228 ± 0.006 | 0.207 ± 0.008 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 1.082 ± 0.001 | 1.082 ± 0.002 | 1.319 ± 0.013 | 1.475 ± 0.022 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 8.542 ± 0.009 | 8.554 ± 0.007 | 10.636 ± 0.052 | 9.979 ± 0.083 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 17.093 ± 0.021 | 17.145 ± 0.009 | 19.616 ± 0.092 | 19.401 ± 0.266 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 4.278 ± 0.011 | 4.292 ± 0.005 | 5.340 ± 0.029 | 5.160 ± 0.046 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 136.062 ± 0.168 | 135.929 ± 0.271 | 199.973 ± 0.789 | 176.722 ± 0.555 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.557 ± 0.002 | 0.552 ± 0.001 | 0.828 ± 0.004 | 0.767 ± 0.002 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 3.017 ± 0.004 | 2.994 ± 0.008 | 4.886 ± 0.023 | 4.030 ± 0.091 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 19.054 ± 0.049 | 18.948 ± 0.031 | 33.874 ± 0.089 | 28.016 ± 0.110 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.134 ± 0.000 | 0.133 ± 0.000 | 0.136 ± 0.001 | 0.126 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 26.677 ± 0.022 | 27.127 ± 0.126 | 47.735 ± 0.113 | 40.313 ± 0.159 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 27.814 ± 0.037 | 28.193 ± 0.083 | 49.841 ± 0.082 | 41.836 ± 0.119 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 31.902 ± 0.043 | 32.030 ± 0.121 | 54.658 ± 0.128 | 46.716 ± 0.203 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.113 ± 0.001 | 0.113 ± 0.000 | 0.176 ± 0.001 | 0.188 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.139 ± 0.001 | 0.130 ± 0.001 | 0.232 ± 0.004 | 0.291 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.241 ± 0.001 | 0.197 ± 0.002 | 0.383 ± 0.006 | 0.458 ± 0.011 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.604 ± 0.001 | 0.610 ± 0.006 | 1.012 ± 0.009 | 0.945 ± 0.021 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.687 ± 0.003 | 0.677 ± 0.005 | 1.175 ± 0.015 | 1.138 ± 0.010 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 1.026 ± 0.002 | 0.928 ± 0.007 | 1.616 ± 0.014 | 1.636 ± 0.020 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 3.813 ± 0.046 | 3.815 ± 0.019 | 6.692 ± 0.146 | 6.801 ± 0.047 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 4.126 ± 0.008 | 4.078 ± 0.012 | 7.284 ± 0.051 | 7.252 ± 0.041 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 5.327 ± 0.015 | 5.168 ± 0.010 | 8.807 ± 0.050 | 8.705 ± 0.031 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.032 ± 0.000 | 0.032 ± 0.000 | 0.055 ± 0.002 | 0.059 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.041 ± 0.001 | 0.036 ± 0.000 | 0.069 ± 0.001 | 0.091 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.078 ± 0.000 | 0.055 ± 0.000 | 0.118 ± 0.001 | 0.168 ± 0.006 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 688.696 ± 1.501 | 667.191 ± 1.220 | 789.145 ± 3.666 | 728.177 ± 4.518 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 3.272 ± 0.009 | 3.273 ± 0.006 | 3.566 ± 0.007 | 3.332 ± 0.016 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 17.931 ± 0.025 | 17.874 ± 0.038 | 19.522 ± 0.032 | 18.446 ± 0.171 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 103.975 ± 0.703 | 103.647 ± 0.369 | 127.045 ± 0.391 | 121.034 ± 0.349 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.831 ± 0.002 | 0.823 ± 0.001 | 0.766 ± 0.005 | 0.750 ± 0.007 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.121 ± 0.000 | 0.119 ± 0.000 | 0.107 ± 0.000 | 0.134 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.167 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.143 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.174 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.147 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.166 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | 0.297 ± 0.004 | 0.081 ± 0.000 | 0.158 ± 0.005 | 0.053 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.227 ± 0.002 | 0.028 ± 0.000 | 0.121 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | 0.485 ± 0.004 | 0.240 ± 0.000 | 0.252 ± 0.006 | 0.141 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.228 ± 0.004 | 0.032 ± 0.000 | 0.124 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.243 ± 0.003 | 0.043 ± 0.000 | 0.132 ± 0.002 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | 0.089 ± 0.001 | 0.081 ± 0.000 | 0.134 ± 0.002 | 0.053 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.057 ± 0.000 | 0.027 ± 0.000 | 0.102 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | 0.185 ± 0.001 | 0.238 ± 0.001 | 0.216 ± 0.005 | 0.142 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.060 ± 0.000 | 0.031 ± 0.000 | 0.105 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.066 ± 0.001 | 0.042 ± 0.000 | 0.111 ± 0.002 | 0.026 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | 0.911 ± 0.008 | 0.045 ± 0.000 | 0.220 ± 0.006 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.873 ± 0.017 | 0.036 ± 0.000 | 0.199 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | 0.959 ± 0.014 | 0.071 ± 0.000 | 0.274 ± 0.005 | 0.051 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.885 ± 0.011 | 0.037 ± 0.000 | 0.196 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.894 ± 0.009 | 0.039 ± 0.000 | 0.201 ± 0.005 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | 0.166 ± 0.001 | 0.044 ± 0.000 | 0.172 ± 0.005 | 0.027 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.152 ± 0.002 | 0.034 ± 0.000 | 0.154 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | 0.208 ± 0.002 | 0.071 ± 0.000 | 0.211 ± 0.005 | 0.051 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.153 ± 0.002 | 0.034 ± 0.000 | 0.153 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.156 ± 0.001 | 0.037 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.047 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.041 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.059 ± 0.001 | 0.040 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.043 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.044 ± 0.001 | 0.024 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | 1.289 ± 0.034 | 0.069 ± 0.000 | 0.158 ± 0.006 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 1.225 ± 0.010 | 0.046 ± 0.000 | 0.139 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | 1.392 ± 0.006 | 0.130 ± 0.000 | 0.199 ± 0.005 | 0.062 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 1.268 ± 0.027 | 0.048 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 1.285 ± 0.005 | 0.052 ± 0.000 | 0.143 ± 0.003 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | 0.149 ± 0.001 | 0.068 ± 0.000 | 0.164 ± 0.006 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.129 ± 0.002 | 0.046 ± 0.000 | 0.142 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | 0.206 ± 0.002 | 0.130 ± 0.000 | 0.202 ± 0.006 | 0.061 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.130 ± 0.001 | 0.047 ± 0.000 | 0.144 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.134 ± 0.001 | 0.051 ± 0.000 | 0.145 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.000 | 0.070 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.064 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.066 ± 0.000 | 0.082 ± 0.001 | 0.038 ± 0.004 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.067 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.067 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | 0.166 ± 0.004 | 0.034 ± 0.000 | 0.332 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.156 ± 0.003 | 0.028 ± 0.000 | 0.322 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | 0.177 ± 0.001 | 0.044 ± 0.000 | 0.350 ± 0.006 | 0.029 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.154 ± 0.002 | 0.029 ± 0.000 | 0.323 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.158 ± 0.003 | 0.030 ± 0.000 | 0.325 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | 0.067 ± 0.001 | 0.034 ± 0.000 | 0.145 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.062 ± 0.000 | 0.028 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | 0.075 ± 0.001 | 0.044 ± 0.000 | 0.164 ± 0.004 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.063 ± 0.001 | 0.029 ± 0.000 | 0.137 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.064 ± 0.001 | 0.030 ± 0.000 | 0.140 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.097 ± 0.000 | 0.107 ± 0.000 | 0.080 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.046 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.309 ± 0.001 | 0.258 ± 0.003 | 0.213 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.052 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.066 ± 0.001 | 0.036 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | 0.312 ± 0.004 | 0.097 ± 0.000 | 0.259 ± 0.008 | 0.078 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.188 ± 0.002 | 0.020 ± 0.000 | 0.183 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | 0.630 ± 0.004 | 0.308 ± 0.000 | 0.432 ± 0.003 | 0.210 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.201 ± 0.004 | 0.026 ± 0.000 | 0.188 ± 0.006 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.219 ± 0.001 | 0.041 ± 0.000 | 0.208 ± 0.005 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | 0.113 ± 0.001 | 0.096 ± 0.000 | 0.175 ± 0.006 | 0.079 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.055 ± 0.001 | 0.020 ± 0.000 | 0.112 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | 0.269 ± 0.001 | 0.308 ± 0.000 | 0.338 ± 0.007 | 0.210 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.059 ± 0.001 | 0.025 ± 0.000 | 0.117 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.070 ± 0.001 | 0.041 ± 0.000 | 0.133 ± 0.001 | 0.034 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.040 ± 0.002 | 0.049 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.039 ± 0.000 | 0.038 ± 0.000 | 0.031 ± 0.000 | 0.046 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.028 ± 0.000 | 0.028 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.029 ± 0.000 | 0.033 ± 0.007 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.062 ± 0.000 | 0.061 ± 0.000 | 0.065 ± 0.000 | 0.073 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.212 ± 0.000 | 0.210 ± 0.000 | 0.209 ± 0.001 | 0.213 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.000 | 0.028 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 24.2x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 22.7x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 7.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 1.922 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.554 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 34.4x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 33.0x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 33.2x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 35.3x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 35.5x faster than the slowest successful cell (`tenferro-eager`, 0.911 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.911 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 62.7x faster than the slowest successful cell (`tenferro-eager`, 0.873 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.873 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`32x32`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.959 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 0.959 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 61.9x faster than the slowest successful cell (`tenferro-eager`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`tenferro-eager`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 42.9x faster than the slowest successful cell (`tenferro-eager`, 0.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 0.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 43.1x faster than the slowest successful cell (`tenferro-eager`, 1.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 1.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 90.6x faster than the slowest successful cell (`tenferro-eager`, 1.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 26.5x faster than the slowest successful cell (`tenferro-eager`, 1.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`32x32`): `jax-cpu` is 22.4x faster than the slowest successful cell (`tenferro-eager`, 1.392 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 1.392 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 87.3x faster than the slowest successful cell (`tenferro-eager`, 1.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`tenferro-eager`, 1.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 62.1x faster than the slowest successful cell (`tenferro-eager`, 1.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 1.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

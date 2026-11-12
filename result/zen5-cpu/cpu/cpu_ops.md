# CPU Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `2ee56deba06eaad06c590db40867fcef00165e86`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_020537/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_020537/run.yaml`
- Timestamp: `2026-11-12T17:05:37.788001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_020537`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_020537/cpu_ops_t1_20261113_020537.csv`

### CPU Benchmark Items

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

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_031144/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_031144/run.yaml`
- Timestamp: `2026-11-12T18:11:44.458524+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_031144`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_031144/cpu_ops_t4_20261113_031144.csv`

### CPU Benchmark Items

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

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261113_035559/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261113_035559/run.yaml`
- Timestamp: `2026-11-12T18:55:59.554485+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261113_035559`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`
- BLAS implementation: `none`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/projects/tensor4all/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261113_035559/cpu_ops_t8_20261113_035559.csv`

### CPU Benchmark Items

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

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

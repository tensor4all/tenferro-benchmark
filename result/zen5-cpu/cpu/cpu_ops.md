# CPU Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`
- Benchmark commit: `3f31f38e0491084b9414035754a3809ef0f5ae94`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_231715/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Timestamp: `20261027_231715`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261027_231715`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261027_231715/cpu_ops_t1_20261027_231715.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_231715/cpu_ops_t1_20261027_231715.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 9.863 ± 0.018 | 9.855 ± 0.005 | 10.465 ± 0.031 | 13.071 ± 0.117 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.149 ± 0.000 | 0.149 ± 0.000 | 0.158 ± 0.000 | 0.211 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.456 ± 0.002 | 2.454 ± 0.005 | 2.632 ± 0.009 | 3.291 ± 0.015 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.602 ± 0.000 | 0.604 ± 0.001 | 0.645 ± 0.002 | 0.830 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.306 ± 0.000 | 0.306 ± 0.001 | 0.193 ± 0.000 | 0.165 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.079 ± 0.000 | 0.079 ± 0.000 | 0.051 ± 0.000 | 0.046 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 1.150 ± 0.001 | 1.148 ± 0.002 | 1.002 ± 0.001 | 0.982 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.290 ± 0.000 | 0.289 ± 0.000 | 0.254 ± 0.000 | 0.250 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.074 ± 0.000 | 0.074 ± 0.000 | 0.066 ± 0.000 | 0.068 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 3.068 ± 0.005 | 3.067 ± 0.001 | 3.039 ± 0.007 | 3.388 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.060 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.768 ± 0.001 | 0.768 ± 0.002 | 0.762 ± 0.001 | 0.854 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.194 ± 0.000 | 0.193 ± 0.000 | 0.193 ± 0.000 | 0.217 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.367 ± 0.004 | 0.364 ± 0.002 | 0.517 ± 0.016 | 0.657 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.068 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.094 ± 0.001 | 0.093 ± 0.001 | 0.125 ± 0.004 | 0.188 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.033 ± 0.001 | 0.091 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.049 ± 0.000 | 0.049 ± 0.000 | 0.017 ± 0.000 | 0.066 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.054 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | 0.062 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.061 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.059 ± 0.001 | 0.058 ± 0.000 | 0.064 ± 0.001 | 0.079 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.060 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.065 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.062 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.102 ± 0.001 | 0.102 ± 0.001 | 0.102 ± 0.001 | 0.149 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.062 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.027 ± 0.001 | 0.027 ± 0.000 | 0.030 ± 0.000 | 0.082 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.067 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 2.070 ± 0.002 | 2.071 ± 0.007 | 2.283 ± 0.011 | 3.309 ± 0.035 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.036 ± 0.000 | 0.034 ± 0.000 | 0.039 ± 0.000 | 0.058 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.526 ± 0.002 | 0.518 ± 0.005 | 0.587 ± 0.005 | 0.823 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.134 ± 0.000 | 0.130 ± 0.001 | 0.150 ± 0.001 | 0.210 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.106 ± 0.000 | 0.105 ± 0.000 | 0.094 ± 0.000 | 0.128 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.029 ± 0.000 | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.268 ± 0.001 | 0.265 ± 0.000 | 0.259 ± 0.001 | 0.367 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.069 ± 0.000 | 0.069 ± 0.000 | 0.067 ± 0.000 | 0.097 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.669 ± 0.002 | 0.664 ± 0.000 | 0.672 ± 0.002 | 1.057 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.171 ± 0.000 | 0.167 ± 0.000 | 0.170 ± 0.001 | 0.269 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.045 ± 0.000 | 0.044 ± 0.000 | 0.045 ± 0.000 | 0.073 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.661 ± 0.003 | 0.691 ± 0.002 | 0.728 ± 0.009 | 1.190 ± 0.025 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.166 ± 0.000 | 0.172 ± 0.001 | 0.187 ± 0.001 | 0.285 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.044 ± 0.000 | 0.044 ± 0.000 | 0.053 ± 0.001 | 0.079 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.091 ± 0.000 | 0.078 ± 0.000 | 0.041 ± 0.000 | 0.129 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.025 ± 0.000 | 0.022 ± 0.000 | 0.016 ± 0.000 | 0.038 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.122 ± 0.000 | 0.092 ± 0.000 | 0.061 ± 0.001 | 0.188 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.033 ± 0.000 | 0.026 ± 0.000 | 0.021 ± 0.000 | 0.053 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.266 ± 0.001 | 0.232 ± 0.001 | 0.213 ± 0.003 | 0.394 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.068 ± 0.000 | 0.059 ± 0.000 | 0.058 ± 0.000 | 0.104 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.033 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 21.182 ± 0.018 | 21.220 ± 0.035 | 22.983 ± 0.039 | 22.971 ± 0.085 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.327 ± 0.001 | 0.327 ± 0.001 | 0.355 ± 0.001 | 0.370 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 5.303 ± 0.029 | 5.308 ± 0.004 | 5.798 ± 0.016 | 5.798 ± 0.030 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 1.321 ± 0.001 | 1.324 ± 0.004 | 1.444 ± 0.003 | 1.454 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.688 ± 0.004 | 0.689 ± 0.002 | 0.536 ± 0.002 | 0.594 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.175 ± 0.001 | 0.174 ± 0.001 | 0.137 ± 0.000 | 0.157 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.046 ± 0.000 | 0.037 ± 0.000 | 0.042 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.384 ± 0.004 | 2.380 ± 0.003 | 2.058 ± 0.004 | 1.927 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.036 ± 0.000 | 0.035 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.599 ± 0.002 | 0.597 ± 0.001 | 0.517 ± 0.001 | 0.486 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.151 ± 0.000 | 0.151 ± 0.000 | 0.132 ± 0.000 | 0.127 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 7.630 ± 0.013 | 7.604 ± 0.005 | 7.054 ± 0.022 | 6.149 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.121 ± 0.000 | 0.120 ± 0.000 | 0.113 ± 0.000 | 0.104 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 1.902 ± 0.006 | 1.901 ± 0.003 | 1.769 ± 0.004 | 1.551 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.478 ± 0.001 | 0.477 ± 0.000 | 0.442 ± 0.001 | 0.391 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.262 ± 0.004 | 4.205 ± 0.059 | 1.545 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.105 ± 0.002 | 0.030 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.312 ± 0.002 | 1.100 ± 0.021 | 0.331 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.082 ± 0.000 | 0.301 ± 0.003 | 0.086 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.148 ± 0.000 | 0.075 ± 0.000 | 0.021 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.040 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.048 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.042 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.183 ± 0.000 | 0.223 ± 0.001 | 0.224 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.042 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.085 ± 0.001 | 0.064 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.051 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.350 ± 0.001 | 2.807 ± 0.019 | 0.417 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.085 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.091 ± 0.000 | 0.734 ± 0.002 | 0.110 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.214 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.135 ± 0.005 | 1.691 ± 0.033 | 1.895 ± 0.026 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.054 ± 0.000 | 0.044 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.292 ± 0.001 | 0.427 ± 0.011 | 0.485 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.078 ± 0.000 | 0.131 ± 0.002 | 0.132 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.143 ± 0.000 | 0.088 ± 0.001 | 0.262 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.029 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.043 ± 0.000 | 0.073 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.032 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.175 ± 0.001 | 0.146 ± 0.001 | 0.375 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.030 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.058 ± 0.001 | 0.101 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.036 ± 0.000 | 0.034 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.522 ± 0.008 | 0.517 ± 0.004 | 0.715 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.036 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.137 ± 0.000 | 0.153 ± 0.001 | 0.186 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.060 ± 0.000 | 0.056 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 152.526 ± 0.196 | 152.507 ± 0.305 | 144.766 ± 0.439 | 124.864 ± 0.305 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 0.672 ± 0.001 | 0.670 ± 0.001 | 0.715 ± 0.001 | 0.734 ± 0.005 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 3.714 ± 0.006 | 3.702 ± 0.006 | 3.817 ± 0.014 | 3.602 ± 0.015 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 23.255 ± 0.164 | 23.152 ± 0.118 | 22.830 ± 0.123 | 20.771 ± 0.073 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.149 ± 0.000 | 0.148 ± 0.000 | 0.155 ± 0.000 | 0.180 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | - | 187.008 ± 0.705 | 258.292 ± 3.675 | 191.724 ± 0.427 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 4.286 ± 0.009 | 5.343 ± 0.010 | 4.439 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 28.003 ± 0.035 | 36.983 ± 0.126 | 28.897 ± 0.042 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | - | 188.994 ± 0.537 | 177.132 ± 0.839 | 190.045 ± 0.245 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 4.295 ± 0.003 | 4.327 ± 0.013 | 4.444 ± 0.010 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 27.836 ± 0.037 | 28.174 ± 0.086 | 28.650 ± 0.097 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | - | 143.633 ± 0.723 | 311.117 ± 0.684 | 144.380 ± 1.637 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 2.654 ± 0.005 | 5.308 ± 0.081 | 2.528 ± 0.004 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 19.780 ± 0.127 | 46.582 ± 0.233 | 18.784 ± 0.037 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | - | 143.876 ± 0.500 | 165.914 ± 1.192 | 144.087 ± 0.727 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 2.672 ± 0.002 | 2.819 ± 0.010 | 2.448 ± 0.003 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 19.376 ± 0.323 | 23.829 ± 0.127 | 18.262 ± 0.043 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 31.122 ± 0.117 | 29.534 ± 0.061 | 34.687 ± 0.142 | 30.636 ± 0.262 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.084 ± 0.001 | 0.068 ± 0.000 | 0.082 ± 0.002 | 0.092 ± 0.006 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 0.510 ± 0.012 | 0.463 ± 0.001 | 0.559 ± 0.006 | 0.570 ± 0.010 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 3.833 ± 0.013 | 3.680 ± 0.004 | 4.708 ± 0.031 | 4.262 ± 0.019 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.020 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.026 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 90.101 ± 0.202 | 98.152 ± 0.362 | 91.268 ± 0.582 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.216 ± 0.001 | 0.255 ± 0.002 | 0.216 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.427 ± 0.005 | 1.574 ± 0.006 | 1.516 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 11.422 ± 0.058 | 13.136 ± 0.045 | 11.823 ± 0.017 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.053 ± 0.001 | 0.041 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | - | 186.426 ± 0.292 | 220.816 ± 2.519 | 199.602 ± 1.597 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 3.672 ± 0.007 | 3.996 ± 0.023 | 3.515 ± 0.004 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 28.479 ± 0.134 | 33.753 ± 0.129 | 26.118 ± 0.064 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | - | 184.472 ± 0.363 | 209.772 ± 2.942 | 197.539 ± 0.428 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 3.610 ± 0.008 | 4.008 ± 0.010 | 3.486 ± 0.007 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 26.629 ± 0.044 | 33.120 ± 0.478 | 26.300 ± 0.050 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 15.387 ± 0.073 | 22.128 ± 0.152 | 16.121 ± 0.044 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.000 | 0.106 ± 0.003 | 0.097 ± 0.005 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.368 ± 0.001 | 0.495 ± 0.008 | 0.421 ± 0.010 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.135 ± 0.008 | 3.470 ± 0.029 | 2.872 ± 0.065 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.045 ± 0.000 | 0.033 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 14.832 ± 0.062 | 22.067 ± 1.072 | 17.103 ± 0.106 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.349 ± 0.004 | 0.533 ± 0.007 | 0.361 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.088 ± 0.007 | 3.175 ± 0.014 | 2.508 ± 0.005 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | - | 14.401 ± 0.065 | 36.808 ± 0.424 | 15.585 ± 0.052 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.350 ± 0.001 | 0.781 ± 0.009 | 0.367 ± 0.002 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.077 ± 0.017 | 5.240 ± 0.008 | 2.512 ± 0.007 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 337.656 ± 1.002 | 363.785 ± 2.996 | 356.682 ± 0.429 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.385 ± 0.003 | 1.560 ± 0.004 | 1.381 ± 0.002 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 7.636 ± 0.016 | 8.341 ± 0.017 | 8.012 ± 0.013 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 52.715 ± 0.307 | 62.563 ± 0.560 | 55.564 ± 0.085 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.298 ± 0.001 | 0.336 ± 0.002 | 0.299 ± 0.002 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | - | 337.576 ± 6.623 | 466.134 ± 1.813 | 351.224 ± 0.688 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 7.617 ± 0.013 | 9.786 ± 0.014 | 7.833 ± 0.016 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 52.643 ± 0.257 | 73.323 ± 0.836 | 55.257 ± 0.106 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | - | 337.412 ± 0.788 | 345.597 ± 2.802 | 358.294 ± 0.440 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 7.606 ± 0.034 | 8.183 ± 0.015 | 7.823 ± 0.011 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 52.158 ± 0.384 | 58.354 ± 0.212 | 55.221 ± 0.160 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 29.340 ± 0.046 | 29.253 ± 0.044 | 31.365 ± 0.173 | 29.747 ± 0.402 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.074 ± 0.000 | 0.067 ± 0.000 | 0.099 ± 0.002 | 0.101 ± 0.008 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.480 ± 0.003 | 0.460 ± 0.002 | 0.626 ± 0.003 | 0.644 ± 0.006 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 3.704 ± 0.031 | 3.669 ± 0.004 | 4.435 ± 0.021 | 4.090 ± 0.111 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 7.394 ± 0.012 | 7.463 ± 0.036 | 8.044 ± 0.008 | 7.686 ± 0.158 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 1.813 ± 0.011 | 1.823 ± 0.002 | 2.213 ± 0.009 | 2.258 ± 0.019 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 65.136 ± 0.055 | 65.191 ± 0.109 | 79.928 ± 0.863 | 68.733 ± 0.389 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.262 ± 0.001 | 0.259 ± 0.001 | 0.322 ± 0.001 | 0.295 ± 0.001 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 1.549 ± 0.003 | 1.537 ± 0.003 | 1.899 ± 0.007 | 1.554 ± 0.009 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 10.832 ± 0.010 | 10.854 ± 0.014 | 13.064 ± 0.025 | 10.990 ± 0.025 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.047 ± 0.000 | 0.045 ± 0.000 | 0.054 ± 0.000 | 0.051 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 13.154 ± 0.123 | 13.549 ± 0.062 | 20.972 ± 0.143 | 16.346 ± 0.039 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 13.896 ± 0.088 | 14.251 ± 0.042 | 22.000 ± 0.187 | 16.926 ± 0.092 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 15.574 ± 0.102 | 15.999 ± 0.045 | 24.162 ± 0.139 | 18.840 ± 0.044 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.059 ± 0.000 | 0.060 ± 0.000 | 0.069 ± 0.000 | 0.075 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.078 ± 0.000 | 0.080 ± 0.001 | 0.092 ± 0.001 | 0.109 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.131 ± 0.000 | 0.135 ± 0.000 | 0.149 ± 0.001 | 0.174 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.314 ± 0.001 | 0.320 ± 0.001 | 0.400 ± 0.003 | 0.376 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.370 ± 0.002 | 0.378 ± 0.002 | 0.458 ± 0.005 | 0.443 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 0.521 ± 0.003 | 0.531 ± 0.001 | 0.635 ± 0.003 | 0.633 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 1.980 ± 0.005 | 2.006 ± 0.018 | 2.703 ± 0.017 | 2.873 ± 0.049 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 2.190 ± 0.009 | 2.205 ± 0.020 | 2.936 ± 0.016 | 3.015 ± 0.056 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 2.693 ± 0.012 | 2.734 ± 0.033 | 3.485 ± 0.024 | 3.660 ± 0.064 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.000 | 0.025 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.027 ± 0.000 | 0.035 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.038 ± 0.000 | 0.039 ± 0.000 | 0.046 ± 0.000 | 0.066 ± 0.001 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 275.704 ± 2.755 | 277.639 ± 1.584 | 316.254 ± 3.059 | 284.585 ± 0.562 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 1.255 ± 0.003 | 1.251 ± 0.004 | 1.380 ± 0.003 | 1.283 ± 0.003 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 6.706 ± 0.046 | 6.718 ± 0.015 | 7.555 ± 0.028 | 7.051 ± 0.014 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 44.111 ± 0.258 | 44.333 ± 0.245 | 48.492 ± 0.127 | 47.033 ± 0.115 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.273 ± 0.001 | 0.274 ± 0.001 | 0.293 ± 0.001 | 0.280 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.038 ± 0.000 | 0.037 ± 0.000 | 0.041 ± 0.000 | 0.052 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.061 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.054 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.063 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.054 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.061 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | - | 0.020 ± 0.000 | 0.060 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.010 ± 0.000 | 0.047 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | - | 0.048 ± 0.000 | 0.094 ± 0.001 | 0.054 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.011 ± 0.000 | 0.048 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.013 ± 0.000 | 0.052 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | - | 0.020 ± 0.000 | 0.052 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.010 ± 0.000 | 0.041 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | - | 0.048 ± 0.000 | 0.082 ± 0.000 | 0.055 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.011 ± 0.000 | 0.041 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.013 ± 0.000 | 0.044 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | - | 0.015 ± 0.000 | 0.083 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.012 ± 0.000 | 0.076 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | - | 0.026 ± 0.000 | 0.104 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.012 ± 0.000 | 0.076 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.013 ± 0.000 | 0.077 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | - | 0.015 ± 0.000 | 0.066 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.012 ± 0.000 | 0.061 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | - | 0.027 ± 0.000 | 0.081 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.012 ± 0.000 | 0.061 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.012 ± 0.000 | 0.062 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.005 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.023 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.005 ± 0.000 | 0.017 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | - | 0.021 ± 0.000 | 0.060 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.000 | 0.054 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | - | 0.035 ± 0.000 | 0.076 ± 0.001 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.000 | 0.055 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | - | 0.022 ± 0.000 | 0.063 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.017 ± 0.000 | 0.056 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | - | 0.036 ± 0.000 | 0.077 ± 0.000 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.000 | 0.056 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.000 | 0.057 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.000 | 0.027 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.025 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.031 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.026 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.026 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.009 ± 0.000 | 0.129 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.013 ± 0.000 | 0.135 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.008 ± 0.000 | 0.124 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | - | 0.009 ± 0.000 | 0.057 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.008 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | - | 0.012 ± 0.000 | 0.064 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.008 ± 0.000 | 0.053 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.008 ± 0.000 | 0.054 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.041 ± 0.000 | 0.031 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.077 ± 0.000 | 0.100 ± 0.000 | 0.083 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.021 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | - | 0.028 ± 0.000 | 0.100 ± 0.001 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.007 ± 0.000 | 0.071 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | - | 0.077 ± 0.000 | 0.166 ± 0.002 | 0.080 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.009 ± 0.000 | 0.072 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.014 ± 0.000 | 0.080 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | - | 0.027 ± 0.000 | 0.069 ± 0.001 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.007 ± 0.000 | 0.044 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | - | 0.077 ± 0.000 | 0.128 ± 0.001 | 0.080 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.008 ± 0.000 | 0.046 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.014 ± 0.000 | 0.052 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.003 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.004 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.023 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.023 ± 0.000 | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.028 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.073 ± 0.001 | 0.071 ± 0.000 | 0.081 ± 0.000 | 0.084 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 19.3x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 25.9x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 24.1x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261027_235042/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Timestamp: `20261027_235042`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261027_235042`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261027_235042/cpu_ops_t4_20261027_235042.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261027_235042/cpu_ops_t4_20261027_235042.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 10.515 ± 0.079 | 10.464 ± 0.012 | 10.416 ± 0.027 | 7.156 ± 0.078 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.149 ± 0.001 | 0.152 ± 0.000 | 0.159 ± 0.000 | 0.169 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.576 ± 0.006 | 2.588 ± 0.003 | 2.687 ± 0.004 | 1.696 ± 0.206 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.617 ± 0.001 | 0.619 ± 0.002 | 0.650 ± 0.002 | 0.520 ± 0.072 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.301 ± 0.000 | 0.302 ± 0.000 | 0.197 ± 0.000 | 0.177 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.078 ± 0.000 | 0.078 ± 0.000 | 0.052 ± 0.000 | 0.046 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.015 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.160 ± 0.001 | 1.159 ± 0.003 | 1.004 ± 0.002 | 0.789 ± 0.115 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.292 ± 0.000 | 0.291 ± 0.000 | 0.249 ± 0.000 | 0.270 ± 0.013 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.075 ± 0.000 | 0.075 ± 0.000 | 0.066 ± 0.000 | 0.071 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 3.202 ± 0.004 | 3.214 ± 0.003 | 3.086 ± 0.007 | 1.643 ± 0.370 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.051 ± 0.000 | 0.051 ± 0.000 | 0.050 ± 0.000 | 0.061 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.786 ± 0.001 | 0.785 ± 0.002 | 0.772 ± 0.001 | 0.651 ± 0.188 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.197 ± 0.000 | 0.197 ± 0.000 | 0.195 ± 0.000 | 0.236 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.378 ± 0.002 | 0.378 ± 0.002 | 0.382 ± 0.012 | 1.359 ± 0.072 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.060 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.096 ± 0.001 | 0.094 ± 0.001 | 0.090 ± 0.001 | 0.232 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.004 | 0.067 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.052 ± 0.000 | 0.052 ± 0.000 | 0.017 ± 0.000 | 0.059 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.056 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | 0.059 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.058 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.061 ± 0.000 | 0.060 ± 0.000 | 0.045 ± 0.015 | 0.132 ± 0.015 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.058 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.000 | 0.059 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.057 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.105 ± 0.000 | 0.104 ± 0.001 | 0.082 ± 0.001 | 0.265 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.058 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.028 ± 0.000 | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.073 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.058 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.175 ± 0.033 | 2.135 ± 0.007 | 2.191 ± 0.052 | 2.164 ± 0.267 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.036 ± 0.001 | 0.035 ± 0.000 | 0.039 ± 0.001 | 0.057 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.538 ± 0.006 | 0.534 ± 0.001 | 0.575 ± 0.006 | 0.634 ± 0.043 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.137 ± 0.000 | 0.135 ± 0.000 | 0.145 ± 0.001 | 0.178 ± 0.024 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.110 ± 0.000 | 0.109 ± 0.000 | 0.084 ± 0.001 | 0.131 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.030 ± 0.000 | 0.030 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.275 ± 0.001 | 0.274 ± 0.000 | 0.247 ± 0.001 | 0.407 ± 0.024 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.072 ± 0.000 | 0.071 ± 0.000 | 0.063 ± 0.000 | 0.099 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.683 ± 0.003 | 0.680 ± 0.001 | 0.660 ± 0.013 | 0.655 ± 0.093 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.172 ± 0.001 | 0.174 ± 0.000 | 0.170 ± 0.001 | 0.312 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.045 ± 0.000 | 0.045 ± 0.000 | 0.072 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.727 ± 0.008 | 0.674 ± 0.011 | 0.348 ± 0.006 | 1.469 ± 0.137 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.027 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.185 ± 0.001 | 0.173 ± 0.001 | 0.120 ± 0.026 | 0.452 ± 0.108 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.049 ± 0.000 | 0.046 ± 0.000 | 0.043 ± 0.007 | 0.104 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.097 ± 0.000 | 0.075 ± 0.000 | 0.035 ± 0.001 | 0.133 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.022 ± 0.000 | 0.015 ± 0.000 | 0.039 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.116 ± 0.001 | 0.095 ± 0.000 | 0.055 ± 0.004 | 0.211 ± 0.015 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.032 ± 0.000 | 0.026 ± 0.000 | 0.019 ± 0.000 | 0.052 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.266 ± 0.002 | 0.239 ± 0.002 | 0.134 ± 0.001 | 0.561 ± 0.103 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.069 ± 0.000 | 0.062 ± 0.000 | 0.052 ± 0.004 | 0.119 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.020 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.033 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 22.578 ± 0.255 | 22.488 ± 0.035 | 23.330 ± 0.420 | 11.671 ± 4.810 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.330 ± 0.001 | 0.335 ± 0.000 | 0.362 ± 0.001 | 0.254 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 5.574 ± 0.014 | 5.574 ± 0.009 | 5.915 ± 0.010 | 2.709 ± 0.270 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 1.350 ± 0.002 | 1.350 ± 0.001 | 1.439 ± 0.010 | 0.920 ± 0.064 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.704 ± 0.002 | 0.708 ± 0.002 | 0.511 ± 0.001 | 0.712 ± 0.076 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.173 ± 0.001 | 0.173 ± 0.001 | 0.129 ± 0.000 | 0.176 ± 0.016 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.045 ± 0.000 | 0.046 ± 0.000 | 0.035 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 2.421 ± 0.003 | 2.425 ± 0.003 | 2.066 ± 0.004 | 1.019 ± 0.234 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.041 ± 0.000 | 0.036 ± 0.000 | 0.036 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.606 ± 0.001 | 0.609 ± 0.001 | 0.520 ± 0.001 | 0.520 ± 0.015 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.152 ± 0.000 | 0.154 ± 0.000 | 0.131 ± 0.000 | 0.133 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 7.981 ± 0.008 | 8.011 ± 0.058 | 7.045 ± 0.006 | 2.808 ± 0.459 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.122 ± 0.001 | 0.124 ± 0.000 | 0.113 ± 0.000 | 0.105 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 1.944 ± 0.002 | 1.948 ± 0.002 | 1.787 ± 0.002 | 0.988 ± 0.113 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.488 ± 0.001 | 0.488 ± 0.001 | 0.450 ± 0.001 | 0.370 ± 0.127 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.324 ± 0.004 | 4.188 ± 0.041 | 2.370 ± 0.403 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.102 ± 0.001 | 0.029 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.331 ± 0.000 | 1.051 ± 0.010 | 0.483 ± 0.141 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.084 ± 0.000 | 0.293 ± 0.003 | 0.102 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.152 ± 0.000 | 0.075 ± 0.000 | 0.018 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.041 ± 0.000 | 0.006 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.049 ± 0.001 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.042 ± 0.000 | 0.008 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.188 ± 0.000 | 0.129 ± 0.008 | 0.335 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.043 ± 0.000 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.086 ± 0.000 | 0.084 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.052 ± 0.000 | 0.026 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.371 ± 0.001 | 2.796 ± 0.011 | 0.741 ± 0.088 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.086 ± 0.000 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.094 ± 0.000 | 0.729 ± 0.005 | 0.151 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.215 ± 0.001 | 0.043 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.109 ± 0.004 | 0.758 ± 0.018 | 2.579 ± 1.038 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.051 ± 0.001 | 0.043 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.289 ± 0.001 | 0.264 ± 0.038 | 0.605 ± 0.109 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.084 ± 0.000 | 0.113 ± 0.005 | 0.162 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.143 ± 0.000 | 0.082 ± 0.002 | 0.304 ± 0.030 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.030 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.043 ± 0.000 | 0.071 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.032 ± 0.000 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.183 ± 0.001 | 0.138 ± 0.001 | 0.421 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.031 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.000 | 0.057 ± 0.001 | 0.104 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.037 ± 0.000 | 0.034 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.525 ± 0.006 | 0.352 ± 0.014 | 0.857 ± 0.091 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.036 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.142 ± 0.001 | 0.140 ± 0.002 | 0.200 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.058 ± 0.001 | 0.052 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 52.239 ± 0.156 | 72.046 ± 63.409 | 173.494 ± 25.617 | 107.516 ± 34.197 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 1.999 ± 0.008 | 0.846 ± 0.001 | 0.647 ± 33.494 | 1.443 ± 32.027 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 2.324 ± 0.005 | 2.315 ± 0.004 | 1.804 ± 88.930 | 11.782 ± 4.102 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 9.133 ± 0.028 | 9.431 ± 0.032 | 14.076 ± 0.043 | 64.720 ± 95.989 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.111 ± 0.000 | 0.110 ± 0.001 | 0.121 ± 0.006 | 0.202 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | - | 65.816 ± 3.257 | 89.330 ± 1.765 | 105.188 ± 6.386 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 11.288 ± 0.060 | 2.371 ± 0.015 | 4.329 ± 0.250 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 31.235 ± 0.298 | 14.159 ± 0.046 | 21.770 ± 2.478 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | - | 68.289 ± 0.484 | 60.836 ± 0.348 | 107.397 ± 5.921 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 5.192 ± 0.017 | 2.055 ± 0.012 | 4.185 ± 0.237 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 11.145 ± 3.104 | 11.005 ± 0.052 | 21.794 ± 3.088 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | - | 49.739 ± 0.263 | 115.178 ± 0.901 | 84.976 ± 6.681 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 3.988 ± 0.017 | 2.229 ± 0.032 | 2.356 ± 0.490 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 9.787 ± 0.070 | 18.516 ± 0.194 | 14.852 ± 2.703 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | - | 51.512 ± 0.269 | 55.290 ± 0.343 | 80.030 ± 6.903 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 2.527 ± 0.015 | 1.082 ± 0.020 | 1.845 ± 0.322 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 7.225 ± 0.090 | 8.289 ± 0.085 | 12.981 ± 2.093 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 18.500 ± 0.448 | 9.685 ± 0.053 | 11.934 ± 0.133 | 9.044 ± 0.230 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.040 ± 0.000 | 0.027 ± 0.000 | 0.060 ± 0.012 | 0.073 ± 0.017 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.632 ± 0.138 | 0.371 ± 0.001 | 0.273 ± 0.033 | 0.368 ± 0.066 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 2.775 ± 0.482 | 1.359 ± 0.005 | 2.687 ± 0.203 | 1.900 ± 0.312 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.009 ± 0.000 | 0.011 ± 0.002 | 0.021 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 28.947 ± 0.197 | 34.022 ± 0.251 | 26.247 ± 1.520 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.090 ± 0.001 | 0.138 ± 0.009 | 0.142 ± 0.014 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.225 ± 0.332 | 0.697 ± 0.057 | 0.656 ± 0.057 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.753 ± 0.084 | 5.702 ± 0.483 | 4.590 ± 0.794 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.045 ± 0.001 | 0.040 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | - | 70.481 ± 39.705 | 73.734 ± 0.357 | 122.938 ± 14.901 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 7.221 ± 0.043 | 1.668 ± 0.008 | 3.324 ± 0.759 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 11.850 ± 0.044 | 12.310 ± 0.039 | 18.769 ± 3.611 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | - | 69.388 ± 1.124 | 72.653 ± 0.361 | 112.952 ± 6.229 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 8.337 ± 1.216 | 1.795 ± 0.037 | 3.417 ± 0.476 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 10.644 ± 0.059 | 13.951 ± 0.067 | 19.929 ± 4.984 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 5.388 ± 0.028 | 15.820 ± 9.857 | 10.721 ± 1.247 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.083 ± 0.000 | 0.109 ± 0.003 | 0.090 ± 0.003 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.386 ± 0.004 | 27.962 ± 1.981 | 2.977 ± 0.035 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.097 ± 0.003 | 29.917 ± 84.974 | 3.422 ± 133.162 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.046 ± 0.000 | 0.033 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 8.963 ± 0.091 | 5.662 ± 0.070 | 9.059 ± 0.308 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.453 ± 0.006 | 0.295 ± 0.068 | 0.298 ± 0.010 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 0.928 ± 0.004 | 1.121 ± 2.991 | 1.635 ± 0.013 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | - | 9.737 ± 0.186 | 10.406 ± 0.089 | 8.167 ± 0.046 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.447 ± 0.003 | 0.344 ± 0.009 | 0.315 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 0.901 ± 0.002 | 1.677 ± 2.995 | 1.663 ± 0.009 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 138.216 ± 0.749 | 163.152 ± 1.577 | 216.289 ± 6.338 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.637 ± 0.232 | 1.322 ± 0.059 | 1.580 ± 0.022 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 10.733 ± 0.138 | 6.443 ± 0.028 | 7.391 ± 0.170 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 26.299 ± 0.598 | 36.495 ± 0.701 | 43.972 ± 2.026 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.286 ± 0.001 | 0.331 ± 0.009 | 0.320 ± 0.263 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | - | 137.813 ± 0.616 | 191.763 ± 1.988 | 214.261 ± 5.431 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 5.343 ± 0.010 | 6.379 ± 0.020 | 7.250 ± 0.131 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 26.240 ± 0.117 | 34.944 ± 0.152 | 43.837 ± 2.596 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | - | 138.654 ± 0.909 | 151.604 ± 1.104 | 215.124 ± 12.100 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 6.016 ± 0.236 | 5.880 ± 0.013 | 7.402 ± 0.099 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 26.587 ± 0.321 | 31.007 ± 0.125 | 43.314 ± 1.597 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 9.160 ± 0.059 | 9.164 ± 0.024 | 10.184 ± 0.112 | 8.511 ± 0.216 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.026 ± 0.000 | 0.023 ± 0.000 | 0.069 ± 0.010 | 0.069 ± 0.009 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.147 ± 0.000 | 0.140 ± 0.001 | 0.327 ± 0.043 | 0.343 ± 0.076 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 1.138 ± 0.003 | 1.209 ± 0.004 | 2.162 ± 0.162 | 1.779 ± 0.237 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 2.321 ± 0.027 | 2.304 ± 0.033 | 3.297 ± 0.262 | 2.794 ± 0.235 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.542 ± 0.005 | 0.543 ± 0.004 | 1.423 ± 0.401 | 1.275 ± 0.318 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 23.604 ± 0.166 | 23.240 ± 0.252 | 26.087 ± 0.237 | 35.896 ± 0.969 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.341 ± 0.000 | 0.647 ± 0.001 | 0.207 ± 0.010 | 0.334 ± 0.003 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.219 ± 0.532 | 1.221 ± 0.002 | 0.935 ± 0.018 | 1.320 ± 0.013 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 4.998 ± 0.021 | 4.703 ± 0.319 | 5.649 ± 0.104 | 7.407 ± 0.263 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.065 ± 0.000 | 0.053 ± 0.000 | 0.060 ± 0.001 | 0.052 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 4.041 ± 0.083 | 8.149 ± 2.944 | 11.897 ± 0.880 | 21.754 ± 136.856 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 4.718 ± 0.040 | 4.901 ± 0.048 | 19.635 ± 1.810 | 11.466 ± 122.622 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 5.184 ± 0.066 | 5.418 ± 0.031 | 13.588 ± 22.935 | 12.290 ± 114.356 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.061 ± 0.000 | 0.496 ± 0.001 | 0.070 ± 0.001 | 0.071 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.097 ± 0.000 | 0.508 ± 0.002 | 9.691 ± 6.996 | 12.517 ± 1.999 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.127 ± 0.001 | 0.526 ± 0.003 | 11.945 ± 2.785 | 1.765 ± 2.053 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.131 ± 0.001 | 0.567 ± 0.001 | 21.984 ± 1.007 | 2.183 ± 32.431 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.161 ± 0.003 | 0.595 ± 0.004 | 32.507 ± 2.987 | 3.299 ± 11.951 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.207 ± 0.001 | 0.863 ± 0.007 | 32.015 ± 1.999 | 1.988 ± 1.228 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 0.637 ± 0.015 | 1.091 ± 0.013 | 108.948 ± 16.968 | 2.472 ± 108.453 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 0.757 ± 0.021 | 1.228 ± 0.025 | 0.985 ± 0.026 | 19.890 ± 125.291 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 0.897 ± 0.014 | 1.803 ± 0.020 | 2.779 ± 111.031 | 18.973 ± 115.392 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.024 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.025 ± 0.001 | 0.041 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.028 ± 0.000 | 0.035 ± 0.000 | 0.055 ± 0.001 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 120.947 ± 1.855 | 120.536 ± 1.864 | 137.008 ± 1.394 | 169.836 ± 2.275 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 1.242 ± 0.005 | 1.441 ± 0.005 | 1.143 ± 0.022 | 1.288 ± 0.005 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 8.010 ± 0.056 | 5.262 ± 0.057 | 5.761 ± 0.021 | 6.031 ± 0.138 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 23.680 ± 0.198 | 23.883 ± 0.207 | 25.482 ± 0.269 | 36.666 ± 0.208 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.265 ± 0.001 | 0.264 ± 0.003 | 0.284 ± 0.002 | 0.306 ± 0.401 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.042 ± 0.000 | 0.041 ± 0.000 | 0.045 ± 0.000 | 0.054 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.057 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.055 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.058 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.055 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.056 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | - | 0.021 ± 0.000 | 0.061 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.012 ± 0.000 | 0.049 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | - | 0.054 ± 0.000 | 0.098 ± 0.001 | 0.052 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.000 | 0.049 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.014 ± 0.000 | 0.052 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | - | 0.021 ± 0.000 | 0.052 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.011 ± 0.000 | 0.041 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | - | 0.054 ± 0.000 | 0.086 ± 0.000 | 0.051 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.012 ± 0.000 | 0.042 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.014 ± 0.000 | 0.045 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | - | 0.017 ± 0.000 | 0.087 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.082 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | - | 0.030 ± 0.000 | 0.108 ± 0.001 | 0.021 ± 0.018 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.014 ± 0.000 | 0.081 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.015 ± 0.000 | 0.083 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | - | 0.016 ± 0.000 | 0.069 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.013 ± 0.000 | 0.065 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | - | 0.037 ± 0.000 | 0.083 ± 0.001 | 0.025 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.013 ± 0.000 | 0.065 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.014 ± 0.000 | 0.066 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.018 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.025 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | - | 0.024 ± 0.000 | 0.064 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.021 ± 0.000 | 0.058 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | - | 0.040 ± 0.000 | 0.078 ± 0.001 | 0.046 ± 0.125 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.021 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.021 ± 0.000 | 0.059 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | - | 0.024 ± 0.000 | 0.067 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.020 ± 0.000 | 0.059 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | - | 0.039 ± 0.000 | 0.080 ± 0.001 | 0.024 ± 0.006 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.020 ± 0.000 | 0.059 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.021 ± 0.000 | 0.061 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.028 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.026 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.032 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.026 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.027 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.010 ± 0.000 | 0.130 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.009 ± 0.000 | 0.129 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.014 ± 0.000 | 0.136 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.009 ± 0.000 | 0.128 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.009 ± 0.000 | 0.130 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | - | 0.010 ± 0.000 | 0.057 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.009 ± 0.000 | 0.056 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | - | 0.014 ± 0.000 | 0.064 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.009 ± 0.000 | 0.055 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.009 ± 0.000 | 0.057 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.042 ± 0.000 | 0.029 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.007 ± 0.000 | 0.019 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.084 ± 0.000 | 0.106 ± 0.001 | 0.083 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.021 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | - | 0.029 ± 0.000 | 0.101 ± 0.001 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.008 ± 0.000 | 0.075 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | - | 0.085 ± 0.000 | 0.172 ± 0.002 | 0.077 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.010 ± 0.000 | 0.074 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.015 ± 0.000 | 0.079 ± 0.009 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | - | 0.028 ± 0.000 | 0.070 ± 0.001 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.008 ± 0.000 | 0.046 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | - | 0.084 ± 0.000 | 0.133 ± 0.001 | 0.077 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.009 ± 0.000 | 0.047 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.014 ± 0.000 | 0.054 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.044 ± 0.000 | 0.008 ± 0.018 | 0.016 ± 0.003 | 0.020 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.025 ± 0.000 | 0.028 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.078 ± 0.000 | 0.076 ± 0.000 | 0.087 ± 0.001 | 0.084 ± 0.009 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 72.5x faster than the slowest successful cell (`pytorch-cpu`, 27.962 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 29.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-eager` is 129.0x faster than the slowest successful cell (`jax-cpu`, 12.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-trace` is 24.6x faster than the slowest successful cell (`jax-cpu`, 12.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-eager` is 94.3x faster than the slowest successful cell (`pytorch-cpu`, 11.945 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-trace` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 11.945 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 21.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-eager` is 168.1x faster than the slowest successful cell (`pytorch-cpu`, 21.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 38.8x faster than the slowest successful cell (`pytorch-cpu`, 21.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-eager` is 202.2x faster than the slowest successful cell (`pytorch-cpu`, 32.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-trace` is 54.6x faster than the slowest successful cell (`pytorch-cpu`, 32.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 32.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-eager` is 155.0x faster than the slowest successful cell (`pytorch-cpu`, 32.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-trace` is 37.1x faster than the slowest successful cell (`pytorch-cpu`, 32.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 44.1x faster than the slowest successful cell (`pytorch-cpu`, 108.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-eager` is 171.1x faster than the slowest successful cell (`pytorch-cpu`, 108.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-trace` is 99.9x faster than the slowest successful cell (`pytorch-cpu`, 108.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `pytorch-cpu` is 20.2x faster than the slowest successful cell (`jax-cpu`, 19.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-eager` is 26.3x faster than the slowest successful cell (`jax-cpu`, 19.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`jax-cpu`, 19.890 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=64`): `tenferro-eager` is 21.1x faster than the slowest successful cell (`jax-cpu`, 18.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=64`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 18.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 20.1x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261028_001612/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Timestamp: `20261028_001612`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261028_001612`.

- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

### CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

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

- tenferro-rs features: `blas-mkl`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.2`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261028_001612/cpu_ops_t8_20261028_001612.csv`
- Source table: `data/results/zen5-cpu/cpu/einsum/20261028_001612/cpu_ops_t8_20261028_001612.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 15.622 ± 0.024 | 15.631 ± 0.015 | 16.334 ± 0.020 | 6.655 ± 1.787 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.237 ± 0.001 | 0.237 ± 0.000 | 0.245 ± 0.001 | 0.247 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 3.907 ± 0.006 | 3.892 ± 0.004 | 4.096 ± 0.009 | 1.620 ± 0.058 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.956 ± 0.001 | 0.961 ± 0.003 | 1.007 ± 0.003 | 0.388 ± 0.024 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.464 ± 0.001 | 0.466 ± 0.001 | 0.305 ± 0.001 | 0.269 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.121 ± 0.000 | 0.120 ± 0.000 | 0.080 ± 0.001 | 0.070 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.034 ± 0.000 | 0.033 ± 0.000 | 0.024 ± 0.000 | 0.021 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 1.784 ± 0.005 | 1.794 ± 0.003 | 1.558 ± 0.003 | 1.150 ± 0.041 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.033 ± 0.000 | 0.032 ± 0.000 | 0.029 ± 0.000 | 0.032 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.452 ± 0.000 | 0.451 ± 0.001 | 0.393 ± 0.001 | 0.424 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.116 ± 0.000 | 0.116 ± 0.000 | 0.102 ± 0.001 | 0.110 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 4.865 ± 0.018 | 4.847 ± 0.004 | 4.737 ± 0.010 | 2.275 ± 0.065 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.080 ± 0.000 | 0.079 ± 0.000 | 0.078 ± 0.000 | 0.095 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 1.217 ± 0.001 | 1.223 ± 0.001 | 1.186 ± 0.002 | 0.677 ± 0.063 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.307 ± 0.001 | 0.307 ± 0.001 | 0.300 ± 0.001 | 0.362 ± 0.018 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 0.589 ± 0.002 | 0.607 ± 0.003 | 0.236 ± 0.009 | 1.382 ± 0.267 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.016 ± 0.001 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.095 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.170 ± 0.001 | 0.149 ± 0.001 | 0.074 ± 0.003 | 0.282 ± 0.021 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.047 ± 0.002 | 0.040 ± 0.000 | 0.029 ± 0.001 | 0.105 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.076 ± 0.000 | 0.077 ± 0.000 | 0.027 ± 0.001 | 0.095 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.089 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.091 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.093 ± 0.001 | 0.092 ± 0.000 | 0.055 ± 0.004 | 0.183 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.091 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.033 ± 0.001 | 0.095 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.161 ± 0.001 | 0.160 ± 0.000 | 0.071 ± 0.003 | 0.364 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.044 ± 0.001 | 0.043 ± 0.000 | 0.027 ± 0.001 | 0.106 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.095 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 3.235 ± 0.007 | 3.237 ± 0.008 | 3.419 ± 0.015 | 1.762 ± 0.056 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.059 ± 0.000 | 0.056 ± 0.000 | 0.062 ± 0.001 | 0.088 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.825 ± 0.007 | 0.819 ± 0.002 | 0.880 ± 0.005 | 0.480 ± 0.039 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.216 ± 0.000 | 0.213 ± 0.002 | 0.230 ± 0.001 | 0.248 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.165 ± 0.000 | 0.169 ± 0.001 | 0.125 ± 0.000 | 0.211 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.045 ± 0.000 | 0.046 ± 0.000 | 0.036 ± 0.000 | 0.055 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.423 ± 0.001 | 0.422 ± 0.001 | 0.378 ± 0.003 | 0.609 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.111 ± 0.000 | 0.108 ± 0.000 | 0.098 ± 0.001 | 0.162 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.000 | 0.031 ± 0.000 | 0.029 ± 0.000 | 0.043 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 1.036 ± 0.002 | 1.034 ± 0.002 | 0.993 ± 0.005 | 0.780 ± 0.103 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.264 ± 0.001 | 0.263 ± 0.001 | 0.258 ± 0.001 | 0.438 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.071 ± 0.000 | 0.069 ± 0.000 | 0.069 ± 0.000 | 0.113 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.093 ± 0.003 | 1.009 ± 0.005 | 0.495 ± 0.019 | 1.662 ± 0.108 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.040 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.280 ± 0.001 | 0.260 ± 0.001 | 0.142 ± 0.007 | 0.450 ± 0.068 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.077 ± 0.000 | 0.072 ± 0.000 | 0.052 ± 0.003 | 0.143 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.147 ± 0.000 | 0.129 ± 0.000 | 0.051 ± 0.000 | 0.230 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.041 ± 0.000 | 0.037 ± 0.000 | 0.023 ± 0.000 | 0.061 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.181 ± 0.001 | 0.154 ± 0.000 | 0.076 ± 0.001 | 0.330 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.050 ± 0.000 | 0.042 ± 0.000 | 0.029 ± 0.000 | 0.088 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.403 ± 0.000 | 0.358 ± 0.017 | 0.196 ± 0.002 | 0.752 ± 0.067 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.107 ± 0.000 | 0.096 ± 0.000 | 0.067 ± 0.002 | 0.178 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.029 ± 0.001 | 0.028 ± 0.000 | 0.027 ± 0.001 | 0.049 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 33.647 ± 0.040 | 33.577 ± 0.047 | 36.181 ± 0.067 | 9.483 ± 0.065 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.520 ± 0.001 | 0.521 ± 0.002 | 0.553 ± 0.002 | 0.252 ± 0.021 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 8.401 ± 0.033 | 8.393 ± 0.005 | 9.028 ± 0.018 | 2.397 ± 0.215 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 2.095 ± 0.008 | 2.100 ± 0.003 | 2.249 ± 0.003 | 0.697 ± 0.046 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 1.050 ± 0.002 | 1.048 ± 0.002 | 0.794 ± 0.002 | 0.999 ± 0.050 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.267 ± 0.001 | 0.266 ± 0.001 | 0.203 ± 0.000 | 0.253 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.071 ± 0.000 | 0.070 ± 0.000 | 0.055 ± 0.000 | 0.067 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 3.736 ± 0.004 | 3.740 ± 0.006 | 3.187 ± 0.003 | 1.279 ± 0.270 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.064 ± 0.000 | 0.063 ± 0.000 | 0.055 ± 0.000 | 0.056 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.934 ± 0.002 | 0.936 ± 0.002 | 0.802 ± 0.001 | 0.824 ± 0.023 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.237 ± 0.001 | 0.237 ± 0.001 | 0.204 ± 0.001 | 0.208 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 12.011 ± 0.010 | 11.994 ± 0.016 | 10.981 ± 0.020 | 2.270 ± 0.471 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.192 ± 0.001 | 0.191 ± 0.001 | 0.175 ± 0.000 | 0.174 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 3.002 ± 0.004 | 3.006 ± 0.005 | 2.755 ± 0.003 | 0.641 ± 0.069 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.754 ± 0.002 | 0.753 ± 0.002 | 0.691 ± 0.001 | 0.491 ± 0.035 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.146 ± 0.013 | 6.033 ± 0.042 | 2.968 ± 0.186 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.159 ± 0.001 | 0.039 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.526 ± 0.008 | 1.605 ± 0.014 | 0.723 ± 0.092 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.139 ± 0.001 | 0.438 ± 0.004 | 0.147 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.232 ± 0.000 | 0.119 ± 0.001 | 0.022 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.063 ± 0.000 | 0.010 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.065 ± 0.000 | 0.076 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.066 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.288 ± 0.001 | 0.181 ± 0.004 | 0.520 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.066 ± 0.000 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.000 | 0.135 ± 0.002 | 0.133 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.080 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.560 ± 0.001 | 4.265 ± 0.026 | 1.069 ± 0.066 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.131 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.144 ± 0.000 | 1.109 ± 0.004 | 0.214 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.328 ± 0.003 | 0.057 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.678 ± 0.006 | 1.149 ± 0.047 | 3.253 ± 0.325 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.078 ± 0.001 | 0.064 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.443 ± 0.014 | 0.358 ± 0.007 | 0.803 ± 0.081 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.130 ± 0.001 | 0.165 ± 0.004 | 0.239 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.233 ± 0.001 | 0.128 ± 0.002 | 0.452 ± 0.012 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.047 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.000 | 0.067 ± 0.001 | 0.120 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.051 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.287 ± 0.001 | 0.209 ± 0.003 | 0.670 ± 0.040 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.048 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.085 ± 0.000 | 0.088 ± 0.002 | 0.172 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.057 ± 0.000 | 0.050 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.801 ± 0.005 | 0.532 ± 0.007 | 1.370 ± 0.112 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.057 ± 0.001 | 0.028 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.223 ± 0.001 | 0.212 ± 0.002 | 0.319 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.000 | 0.089 ± 0.001 | 0.086 ± 0.004 | - | - |
| large | `eigh` | f64 | 8 | `1024x1024` | - | 128.201 ± 3.816 | 55.041 ± 0.156 | 176.905 ± 18.255 | 214.049 ± 78.815 | - | - |
| large | `eigh` | f64 | 8 | `128x128` | - | 0.958 ± 0.003 | 0.959 ± 0.002 | 2.066 ± 65.982 | 37.887 ± 0.754 | - | - |
| large | `eigh` | f64 | 8 | `256x256` | - | 18.800 ± 0.056 | 11.581 ± 0.132 | 76.439 ± 8.497 | 9.824 ± 142.022 | - | - |
| large | `eigh` | f64 | 8 | `512x512` | - | 33.323 ± 0.300 | 33.022 ± 0.032 | 152.904 ± 120.860 | 151.430 ± 17.376 | - | - |
| large | `eigh` | f64 | 8 | `64x64` | - | 0.167 ± 0.001 | 0.166 ± 0.000 | 0.177 ± 0.003 | 0.292 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | - | - | 145.846 ± 71.297 | 88.704 ± 0.388 | 131.017 ± 14.524 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | - | - | 23.445 ± 0.233 | 2.969 ± 0.032 | 7.389 ± 0.450 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | - | - | 39.714 ± 16.619 | 13.546 ± 0.278 | 33.056 ± 3.104 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | - | - | 70.888 ± 62.232 | 61.875 ± 0.340 | 127.451 ± 7.475 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | - | - | 23.683 ± 0.036 | 2.624 ± 0.024 | 7.439 ± 0.551 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | - | - | 56.452 ± 0.163 | 10.593 ± 0.061 | 30.960 ± 2.552 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | - | - | 72.732 ± 7.440 | 105.859 ± 1.208 | 78.793 ± 23.053 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | - | - | 3.673 ± 0.256 | 2.608 ± 0.035 | 3.972 ± 0.802 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | - | - | 13.972 ± 0.074 | 17.580 ± 0.324 | 20.051 ± 7.398 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | - | - | 80.207 ± 2.773 | 52.381 ± 0.437 | 87.642 ± 10.241 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | - | - | 5.358 ± 0.448 | 1.352 ± 0.030 | 3.727 ± 0.760 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | - | - | 13.830 ± 0.152 | 8.620 ± 0.095 | 16.255 ± 4.643 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `1024x1024` | - | 15.598 ± 0.128 | 8.951 ± 0.071 | 10.513 ± 0.100 | 7.422 ± 0.911 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `128x128` | - | 0.049 ± 0.001 | 0.026 ± 0.000 | 0.050 ± 0.005 | 0.083 ± 0.005 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `256x256` | - | 0.890 ± 0.043 | 0.806 ± 0.001 | 0.219 ± 0.023 | 0.308 ± 0.028 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `512x512` | - | 2.765 ± 0.117 | 1.684 ± 0.013 | 1.909 ± 0.215 | 1.705 ± 0.171 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `64x64` | - | 0.022 ± 0.000 | 0.066 ± 0.000 | 0.017 ± 0.002 | 0.029 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 26.860 ± 0.126 | 30.794 ± 0.231 | 21.980 ± 0.690 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.090 ± 0.000 | 0.133 ± 0.007 | 0.157 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.126 ± 0.002 | 0.559 ± 0.020 | 0.605 ± 0.037 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 4.139 ± 0.014 | 4.741 ± 0.154 | 3.780 ± 0.223 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.197 ± 0.157 | 0.057 ± 0.001 | 0.056 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | - | - | 72.724 ± 0.951 | 75.625 ± 0.350 | 130.987 ± 18.173 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | - | - | 11.764 ± 0.257 | 2.129 ± 0.033 | 6.683 ± 1.141 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | - | - | 26.142 ± 0.685 | 11.916 ± 0.041 | 27.349 ± 5.019 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | - | - | 162.127 ± 60.281 | 72.048 ± 0.613 | 126.977 ± 16.489 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | - | - | 12.632 ± 0.378 | 2.294 ± 0.018 | 6.924 ± 1.097 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | - | - | 21.594 ± 0.283 | 13.993 ± 0.229 | 28.212 ± 4.203 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `1024x1024,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 20.034 ± 44.942 | 99.863 ± 109.464 | 135.058 ± 32.207 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `128x128,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.128 ± 0.000 | 0.170 ± 0.003 | 0.156 ± 0.005 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `256x256,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.337 ± 0.003 | 27.940 ± 2.983 | 1.140 ± 0.341 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `512x512,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 3.071 ± 0.009 | 32.841 ± 11.578 | 2.601 ± 131.678 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.072 ± 0.001 | 0.047 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | - | - | 32.936 ± 33.342 | 6.222 ± 0.079 | 9.895 ± 0.620 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | - | - | 1.771 ± 0.004 | 0.921 ± 1.365 | 0.348 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | - | - | 2.128 ± 0.027 | 1.411 ± 0.056 | 2.285 ± 0.871 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | - | - | 15.299 ± 26.607 | 10.180 ± 0.209 | 8.984 ± 0.301 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | - | - | 1.767 ± 0.004 | 0.553 ± 0.024 | 0.354 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | - | - | 2.643 ± 0.009 | 1.984 ± 0.012 | 2.241 ± 0.063 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `1024x1024` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 155.048 ± 1.790 | 185.431 ± 1.680 | 285.891 ± 8.772 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `128x128` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 2.603 ± 0.011 | 1.858 ± 0.015 | 2.509 ± 0.077 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `256x256` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 8.659 ± 0.068 | 8.693 ± 0.040 | 11.951 ± 0.991 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `512x512` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 33.624 ± 0.490 | 41.858 ± 0.380 | 65.127 ± 7.355 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.433 ± 0.001 | 0.482 ± 0.002 | 0.483 ± 0.004 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | - | - | 153.806 ± 1.073 | 212.593 ± 3.732 | 286.415 ± 13.652 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | - | - | 14.975 ± 0.191 | 9.340 ± 0.056 | 12.494 ± 0.835 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | - | - | 61.152 ± 20.178 | 42.266 ± 0.200 | 64.028 ± 7.731 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | - | - | 154.665 ± 1.333 | 176.369 ± 4.091 | 293.920 ± 13.207 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | - | - | 15.362 ± 0.549 | 8.582 ± 0.029 | 11.971 ± 0.956 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | - | - | 61.142 ± 0.590 | 40.990 ± 1.113 | 67.046 ± 3.091 | - | - |
| large | `matmul` | f64 | 8 | `1024x1024` | - | 8.196 ± 0.062 | 8.053 ± 0.070 | 9.191 ± 0.129 | 7.646 ± 0.863 | - | - |
| large | `matmul` | f64 | 8 | `128x128` | - | 0.031 ± 0.000 | 0.022 ± 0.000 | 0.057 ± 0.005 | 0.063 ± 0.004 | - | - |
| large | `matmul` | f64 | 8 | `256x256` | - | 0.140 ± 0.004 | 0.112 ± 0.001 | 0.246 ± 0.016 | 0.291 ± 0.025 | - | - |
| large | `matmul` | f64 | 8 | `512x512` | - | 0.957 ± 0.009 | 0.957 ± 0.007 | 1.650 ± 0.177 | 1.508 ± 0.141 | - | - |
| large | `matmul_rect` | f64 | 8 | `1024x256 * 256x1024` | - | 2.176 ± 0.020 | 2.208 ± 0.072 | 2.603 ± 0.124 | 2.433 ± 0.247 | - | - |
| large | `matmul_rect` | f64 | 8 | `256x1024 * 1024x256` | - | 0.427 ± 0.003 | 0.433 ± 0.002 | 0.872 ± 0.101 | 1.021 ± 0.079 | - | - |
| large | `qr` | f64 | 8 | `1024x1024` | - | 42.365 ± 1.530 | 27.179 ± 0.089 | 30.690 ± 3.664 | 45.476 ± 1.384 | - | - |
| large | `qr` | f64 | 8 | `128x128` | - | 0.587 ± 0.001 | 0.573 ± 0.002 | 0.280 ± 0.003 | 0.531 ± 0.010 | - | - |
| large | `qr` | f64 | 8 | `256x256` | - | 1.860 ± 0.110 | 2.473 ± 0.021 | 1.183 ± 0.009 | 2.058 ± 0.040 | - | - |
| large | `qr` | f64 | 8 | `512x512` | - | 8.951 ± 0.179 | 8.447 ± 3.232 | 5.967 ± 0.229 | 10.365 ± 0.083 | - | - |
| large | `qr` | f64 | 8 | `64x64` | - | 0.108 ± 0.000 | 0.108 ± 0.000 | 0.102 ± 0.002 | 0.080 ± 0.003 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=1` | - | 3.809 ± 0.074 | 3.927 ± 0.050 | 77.950 ± 104.084 | 134.573 ± 32.610 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=16` | - | 4.211 ± 0.040 | 32.299 ± 26.746 | 116.570 ± 68.032 | 149.175 ± 30.835 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=64` | - | 6.718 ± 0.040 | 12.696 ± 11.560 | 98.047 ± 51.445 | 136.933 ± 16.986 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=1` | - | 0.098 ± 0.001 | 0.098 ± 0.000 | 0.110 ± 0.001 | 0.124 ± 0.009 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=16` | - | 0.112 ± 0.002 | 0.384 ± 0.002 | 8.314 ± 0.012 | 14.987 ± 2.999 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=64` | - | 0.127 ± 0.002 | 0.397 ± 0.002 | 7.109 ± 0.018 | 4.009 ± 0.070 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=1` | - | 0.341 ± 0.003 | 1.024 ± 0.003 | 21.977 ± 3.009 | 1.145 ± 1.273 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=16` | - | 0.416 ± 0.005 | 1.145 ± 0.050 | 33.196 ± 3.000 | 6.649 ± 65.899 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=64` | - | 0.495 ± 0.007 | 1.223 ± 0.003 | 31.792 ± 1.003 | 12.270 ± 105.436 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=1` | - | 0.927 ± 0.024 | 1.634 ± 0.032 | 86.898 ± 57.993 | 131.230 ± 139.617 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=16` | - | 0.871 ± 0.023 | 1.851 ± 0.026 | 46.236 ± 47.981 | 126.248 ± 143.138 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=64` | - | 1.159 ± 0.045 | 3.039 ± 0.061 | 36.520 ± 24.935 | 11.964 ± 122.992 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.033 ± 0.001 | 0.036 ± 0.002 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=16` | - | 0.033 ± 0.000 | 0.035 ± 0.000 | 0.039 ± 0.001 | 0.059 ± 0.007 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=64` | - | 0.038 ± 0.001 | 0.041 ± 0.001 | 0.047 ± 0.001 | 0.077 ± 0.002 | - | - |
| large | `svd` | f64 | 8 | `1024x1024` | - | 139.595 ± 38.645 | 146.365 ± 29.766 | 168.432 ± 3.890 | 235.705 ± 2.948 | - | - |
| large | `svd` | f64 | 8 | `128x128` | - | 1.855 ± 0.012 | 1.854 ± 0.008 | 1.707 ± 0.005 | 2.011 ± 0.007 | - | - |
| large | `svd` | f64 | 8 | `256x256` | - | 7.709 ± 0.093 | 11.293 ± 0.033 | 8.140 ± 0.023 | 9.419 ± 0.066 | - | - |
| large | `svd` | f64 | 8 | `512x512` | - | 74.729 ± 18.092 | 75.925 ± 2.843 | 33.397 ± 0.134 | 53.530 ± 0.451 | - | - |
| large | `svd` | f64 | 8 | `64x64` | - | 0.405 ± 0.001 | 0.405 ± 0.002 | 0.426 ± 0.003 | 0.451 ± 0.003 | - | - |
| small | `eigh` | f64 | 8 | `16x16` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `32x32` | - | 0.066 ± 0.001 | 0.065 ± 0.000 | 0.068 ± 0.000 | 0.083 ± 0.004 | - | - |
| small | `eigh` | f64 | 8 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `16x16` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.090 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `2x2` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.087 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `32x32` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.091 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.088 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `8x8` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.090 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | - | - | 0.034 ± 0.000 | 0.095 ± 0.001 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | - | - | 0.018 ± 0.000 | 0.075 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | - | - | 0.085 ± 0.000 | 0.155 ± 0.003 | 0.084 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | - | - | 0.020 ± 0.000 | 0.076 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | - | - | 0.023 ± 0.000 | 0.080 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | - | - | 0.034 ± 0.000 | 0.081 ± 0.001 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | - | - | 0.018 ± 0.000 | 0.063 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | - | - | 0.085 ± 0.000 | 0.135 ± 0.002 | 0.080 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | - | - | 0.019 ± 0.000 | 0.064 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | - | - | 0.022 ± 0.000 | 0.067 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | - | - | 0.027 ± 0.000 | 0.138 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | - | - | 0.022 ± 0.000 | 0.127 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | - | - | 0.051 ± 0.000 | 0.209 ± 0.003 | 0.061 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | - | - | 0.023 ± 0.000 | 0.125 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | - | - | 0.023 ± 0.000 | 0.127 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | - | - | 0.026 ± 0.000 | 0.108 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | - | - | 0.021 ± 0.000 | 0.099 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | - | - | 0.061 ± 0.000 | 0.135 ± 0.002 | 0.034 ± 0.126 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | - | - | 0.021 ± 0.000 | 0.099 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | - | - | 0.022 ± 0.000 | 0.101 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.028 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.025 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.039 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.009 ± 0.000 | 0.026 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.026 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | - | - | 0.038 ± 0.000 | 0.099 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | - | - | 0.032 ± 0.000 | 0.088 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | - | - | 0.062 ± 0.000 | 0.130 ± 0.001 | 0.249 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | - | - | 0.032 ± 0.000 | 0.087 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | - | - | 0.034 ± 0.000 | 0.090 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | - | - | 0.038 ± 0.000 | 0.102 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | - | - | 0.031 ± 0.000 | 0.090 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | - | - | 0.061 ± 0.000 | 0.142 ± 0.001 | 0.046 ± 0.066 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | - | - | 0.032 ± 0.000 | 0.090 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | - | - | 0.033 ± 0.000 | 0.093 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `16x16,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.042 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.039 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `32x32,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.049 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.040 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.041 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | - | - | 0.016 ± 0.000 | 0.206 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | - | - | 0.014 ± 0.000 | 0.200 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | - | - | 0.022 ± 0.000 | 0.215 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | - | - | 0.015 ± 0.000 | 0.200 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | - | - | 0.015 ± 0.000 | 0.202 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | - | - | 0.016 ± 0.000 | 0.089 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | - | - | 0.014 ± 0.000 | 0.084 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | - | - | 0.022 ± 0.000 | 0.102 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | - | - | 0.014 ± 0.000 | 0.084 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | - | - | 0.015 ± 0.000 | 0.086 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `16x16` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.065 ± 0.000 | 0.046 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.029 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `32x32` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.133 ± 0.000 | 0.162 ± 0.001 | 0.130 ± 0.004 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.032 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.041 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | - | - | 0.045 ± 0.000 | 0.159 ± 0.005 | 0.041 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | - | - | 0.013 ± 0.000 | 0.113 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | - | - | 0.134 ± 0.000 | 0.271 ± 0.004 | 0.123 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | - | - | 0.015 ± 0.000 | 0.115 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | - | - | 0.024 ± 0.000 | 0.127 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | - | - | 0.045 ± 0.000 | 0.108 ± 0.001 | 0.042 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | - | - | 0.012 ± 0.000 | 0.069 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | - | - | 0.133 ± 0.000 | 0.208 ± 0.003 | 0.122 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | - | - | 0.014 ± 0.000 | 0.071 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | - | - | 0.023 ± 0.000 | 0.081 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `16x16` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `32x32` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `64x64` | - | 0.066 ± 0.318 | 0.042 ± 0.023 | 0.018 ± 0.001 | 0.028 ± 0.004 | - | - |
| small | `matmul` | f64 | 8 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `16x16` | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `32x32` | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.025 ± 0.001 | - | - |
| small | `qr` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=1` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=4` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `16x16` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.039 ± 0.000 | 0.044 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `32x32` | - | 0.123 ± 0.001 | 0.122 ± 0.000 | 0.133 ± 0.001 | 0.128 ± 0.004 | - | - |
| small | `svd` | f64 | 8 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `pytorch-cpu` is 18.3x faster than the slowest successful cell (`jax-cpu`, 37.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `tenferro-eager` is 39.5x faster than the slowest successful cell (`jax-cpu`, 37.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`128x128`): `tenferro-trace` is 39.5x faster than the slowest successful cell (`jax-cpu`, 37.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 27.940 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 27.940 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 32.841 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 32.841 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-eager` is 35.3x faster than the slowest successful cell (`jax-cpu`, 134.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-trace` is 34.3x faster than the slowest successful cell (`jax-cpu`, 134.573 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-eager` is 35.4x faster than the slowest successful cell (`jax-cpu`, 149.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-eager` is 20.4x faster than the slowest successful cell (`jax-cpu`, 136.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 136.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-eager` is 133.3x faster than the slowest successful cell (`jax-cpu`, 14.987 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-trace` is 39.0x faster than the slowest successful cell (`jax-cpu`, 14.987 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-eager` is 56.0x faster than the slowest successful cell (`pytorch-cpu`, 7.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 7.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 21.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-eager` is 64.5x faster than the slowest successful cell (`pytorch-cpu`, 21.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 21.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-eager` is 79.7x faster than the slowest successful cell (`pytorch-cpu`, 33.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-trace` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 33.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-eager` is 64.2x faster than the slowest successful cell (`pytorch-cpu`, 31.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`pytorch-cpu`, 31.792 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-eager` is 141.6x faster than the slowest successful cell (`jax-cpu`, 131.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 80.3x faster than the slowest successful cell (`jax-cpu`, 131.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-eager` is 144.9x faster than the slowest successful cell (`jax-cpu`, 126.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-trace` is 68.2x faster than the slowest successful cell (`jax-cpu`, 126.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-eager` is 31.5x faster than the slowest successful cell (`pytorch-cpu`, 36.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 36.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-eager` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-eager` is 24.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-eager` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-eager` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`32x32,rhs=1`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `jax-cpu` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `jax-cpu` is 22.2x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`
- Benchmark commit: `320964928fdd481aaaf81ceae9606c0ea8df559f`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261006_172037/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261006_172037`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261006_172037`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/t4a-2010-bench-migrate/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261006_172037/cpu_ops_t1_20261006_172037.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261006_172037/cpu_ops_t1_20261006_172037.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.014 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.045 ± 0.000 | 0.035 ± 0.001 | 0.066 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.001 | 0.038 ± 0.000 | 0.034 ± 0.000 | 0.036 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.123 ± 0.000 | 0.126 ± 0.001 | 0.112 ± 0.002 | 0.268 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.175 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.155 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.166 ± 0.003 | 0.124 ± 0.000 | 0.149 ± 0.002 | 0.156 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.156 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.011 ± 0.001 | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.157 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.001 | 0.021 ± 0.000 | 0.017 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.036 ± 0.000 | 0.038 ± 0.000 | 0.034 ± 0.000 | 0.123 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.002 | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.044 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.004 | 0.014 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.000 | 0.061 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.025 ± 0.001 | 0.028 ± 0.001 | 0.024 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.079 ± 0.000 | 0.082 ± 0.000 | 0.067 ± 0.000 | 0.081 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.069 ± 0.000 | 0.073 ± 0.000 | 0.064 ± 0.000 | 0.065 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.254 ± 0.010 | 0.258 ± 0.004 | 0.225 ± 0.000 | 0.523 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.215 ± 0.026 | 0.035 ± 0.000 | 0.121 ± 0.000 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.228 ± 0.002 | 0.050 ± 0.002 | 0.124 ± 0.001 | 0.034 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.218 ± 0.001 | 0.037 ± 0.000 | 0.126 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.234 ± 0.001 | 0.055 ± 0.000 | 0.146 ± 0.003 | 0.072 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.286 ± 0.001 | 0.048 ± 0.000 | 0.114 ± 0.001 | 0.020 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.296 ± 0.029 | 0.058 ± 0.000 | 0.121 ± 0.006 | 0.082 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.289 ± 0.007 | 0.049 ± 0.000 | 0.117 ± 0.000 | 0.046 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.279 ± 0.003 | 0.064 ± 0.000 | 0.128 ± 0.007 | 0.114 ± 0.003 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.262 ± 0.001 | 0.262 ± 0.000 | 0.274 ± 0.011 | 0.578 ± 0.106 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 7.537 ± 0.014 | 8.831 ± 0.011 | 8.864 ± 0.755 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 47.268 ± 0.449 | 57.227 ± 0.833 | 47.633 ± 2.472 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 7.491 ± 0.084 | 7.290 ± 0.019 | 8.705 ± 0.032 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 46.895 ± 0.079 | 44.833 ± 0.345 | 45.792 ± 1.836 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 5.255 ± 0.398 | 10.846 ± 0.047 | 4.023 ± 0.023 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 29.789 ± 0.520 | 71.088 ± 0.637 | 24.310 ± 0.559 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 5.194 ± 0.011 | 4.649 ± 0.035 | 4.220 ± 0.026 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 29.331 ± 0.237 | 37.136 ± 0.473 | 28.105 ± 4.177 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.035 ± 0.000 | 0.029 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.275 ± 0.006 | 0.090 ± 0.000 | 0.119 ± 0.005 | 0.056 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 6.471 ± 0.006 | 6.279 ± 0.013 | 6.732 ± 0.035 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 46.085 ± 0.505 | 44.571 ± 0.998 | 43.010 ± 4.969 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 6.398 ± 0.065 | 6.281 ± 0.012 | 6.852 ± 0.032 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 43.404 ± 0.282 | 44.288 ± 0.363 | 44.435 ± 1.604 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.322 ± 0.015 | 0.077 ± 0.000 | 0.143 ± 0.000 | 0.052 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.657 ± 0.002 | 1.072 ± 0.011 | 1.422 ± 0.048 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 3.627 ± 0.026 | 4.805 ± 0.069 | 4.280 ± 0.022 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.640 ± 0.002 | 1.471 ± 0.001 | 1.416 ± 0.016 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 3.551 ± 0.117 | 8.151 ± 0.020 | 4.235 ± 0.028 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.798 ± 0.002 | 0.585 ± 0.003 | 0.664 ± 0.001 | 1.410 ± 0.301 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 13.043 ± 0.547 | 18.086 ± 0.047 | 13.146 ± 0.674 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 89.121 ± 4.801 | 118.288 ± 1.182 | 90.660 ± 1.320 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 12.978 ± 0.426 | 13.594 ± 0.113 | 12.946 ± 0.938 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 87.677 ± 2.333 | 93.180 ± 0.806 | 97.408 ± 3.399 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.164 ± 0.001 | 0.124 ± 0.000 | 0.162 ± 0.012 | 0.325 ± 0.018 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.920 ± 0.000 | 0.756 ± 0.001 | 0.900 ± 0.040 | 1.669 ± 0.046 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 2.883 ± 0.006 | 2.960 ± 0.002 | 2.999 ± 0.017 | 7.010 ± 0.216 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.086 ± 0.000 | 0.088 ± 0.000 | 0.094 ± 0.000 | 0.215 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.034 ± 0.000 | 0.042 ± 0.000 | 0.041 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.040 ± 0.001 | 0.045 ± 0.001 | 0.057 ± 0.001 | 0.071 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.072 ± 0.001 | 0.077 ± 0.000 | 0.105 ± 0.006 | 0.171 ± 0.005 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.507 ± 0.001 | 0.505 ± 0.004 | 0.536 ± 0.003 | 1.268 ± 0.024 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.143 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.144 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.151 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.053 ± 0.000 | 0.167 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.059 ± 0.000 | 0.171 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.062 ± 0.000 | 0.178 ± 0.016 | 0.046 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.055 ± 0.002 | 0.167 ± 0.008 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.061 ± 0.005 | 0.170 ± 0.003 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.065 ± 0.002 | 0.210 ± 0.004 | 0.049 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.053 ± 0.010 | 0.234 ± 0.009 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.054 ± 0.006 | 0.233 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.055 ± 0.000 | 0.247 ± 0.046 | 0.049 ± 0.007 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.055 ± 0.000 | 0.221 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.057 ± 0.000 | 0.219 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.057 ± 0.000 | 0.221 ± 0.005 | 0.052 ± 0.006 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.196 ± 0.001 | 0.028 ± 0.001 | 0.062 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.197 ± 0.001 | 0.028 ± 0.001 | 0.062 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.199 ± 0.000 | 0.029 ± 0.000 | 0.062 ± 0.000 | 0.032 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.093 ± 0.002 | 0.186 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.094 ± 0.000 | 0.186 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.095 ± 0.000 | 0.188 ± 0.003 | 0.048 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.094 ± 0.000 | 0.209 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.097 ± 0.014 | 0.211 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.099 ± 0.001 | 0.214 ± 0.003 | 0.042 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.275 ± 0.034 | 0.040 ± 0.000 | 0.103 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.271 ± 0.022 | 0.041 ± 0.006 | 0.104 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.274 ± 0.000 | 0.042 ± 0.000 | 0.105 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.030 ± 0.000 | 0.347 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.031 ± 0.000 | 0.348 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.031 ± 0.000 | 0.350 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.031 ± 0.000 | 0.209 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.032 ± 0.000 | 0.211 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.032 ± 0.000 | 0.211 ± 0.003 | 0.016 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.241 ± 0.002 | 0.031 ± 0.001 | 0.094 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.242 ± 0.001 | 0.037 ± 0.000 | 0.098 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.252 ± 0.002 | 0.046 ± 0.000 | 0.107 ± 0.000 | 0.044 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.000 | 0.228 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.036 ± 0.000 | 0.243 ± 0.039 | 0.016 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.046 ± 0.000 | 0.242 ± 0.002 | 0.045 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.032 ± 0.000 | 0.191 ± 0.032 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.037 ± 0.000 | 0.186 ± 0.002 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.046 ± 0.000 | 0.196 ± 0.003 | 0.053 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.034 ± 0.002 | 0.026 ± 0.000 | 0.033 ± 0.001 | 0.065 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.016 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.018 ± 0.000 | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 22.4x faster than the slowest successful cell (`tenferro-eager`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 43.2x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 40.7x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 42.4x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.234 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.348 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.348 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.3x faster than the slowest successful cell (`tenferro-eager`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.228 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261006_175113/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261006_175113`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261006_175113`.

- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.1`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/t4a-2010-bench-migrate/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261006_175113/cpu_ops_t4_20261006_175113.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261006_175113/cpu_ops_t4_20261006_175113.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.002 | 0.018 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.049 ± 0.000 | 0.051 ± 0.000 | 0.040 ± 0.001 | 0.069 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.000 | 0.045 ± 0.000 | 0.040 ± 0.000 | 0.036 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.145 ± 0.000 | 0.148 ± 0.000 | 0.118 ± 0.013 | 0.270 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.012 ± 0.001 | 0.185 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.024 ± 0.000 | 0.015 ± 0.001 | 0.188 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.137 ± 0.003 | 0.138 ± 0.000 | 0.117 ± 0.018 | 0.284 ± 0.169 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.001 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.188 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.025 ± 0.001 | 0.020 ± 0.001 | 0.185 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.002 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.025 ± 0.000 | 0.031 ± 0.001 | 0.028 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.020 ± 0.000 | 0.022 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.048 ± 0.000 | 0.062 ± 0.002 | 0.081 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.004 | 0.014 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.030 ± 0.003 | 0.043 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.001 | 0.013 ± 0.000 | 0.027 ± 0.003 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.001 | 0.018 ± 0.000 | 0.036 ± 0.002 | 0.061 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.033 ± 0.000 | 0.024 ± 0.004 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.000 | 0.093 ± 0.000 | 0.077 ± 0.003 | 0.093 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.082 ± 0.000 | 0.086 ± 0.000 | 0.076 ± 0.000 | 0.078 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.000 | 0.304 ± 0.000 | 0.225 ± 0.019 | 0.523 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.275 ± 0.003 | 0.061 ± 0.000 | 0.121 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.299 ± 0.002 | 0.078 ± 0.000 | 0.127 ± 0.024 | 0.039 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.265 ± 0.026 | 0.065 ± 0.001 | 0.150 ± 0.002 | 0.030 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.001 | 0.086 ± 0.000 | 0.148 ± 0.010 | 0.082 ± 0.022 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.324 ± 0.002 | 0.065 ± 0.000 | 0.141 ± 0.018 | 0.024 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.341 ± 0.003 | 0.078 ± 0.000 | 0.149 ± 0.009 | 0.088 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.333 ± 0.007 | 0.067 ± 0.001 | 0.145 ± 0.001 | 0.040 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.345 ± 0.000 | 0.084 ± 0.000 | 0.160 ± 0.002 | 0.071 ± 0.004 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.389 ± 0.053 | 0.316 ± 0.001 | 0.311 ± 0.009 | 0.638 ± 0.069 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 6.588 ± 0.099 | 6.623 ± 0.148 | 13.844 ± 0.071 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 26.779 ± 2.387 | 32.868 ± 0.345 | 47.635 ± 11.162 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 6.632 ± 0.020 | 5.690 ± 0.130 | 13.812 ± 0.154 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 26.513 ± 0.592 | 24.135 ± 0.977 | 48.849 ± 4.550 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 3.243 ± 0.290 | 7.269 ± 0.924 | 3.156 ± 0.259 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 16.173 ± 1.167 | 35.631 ± 0.763 | 17.851 ± 3.369 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 4.061 ± 0.122 | 3.148 ± 0.034 | 3.619 ± 0.319 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 15.981 ± 0.531 | 17.956 ± 0.406 | 17.787 ± 6.863 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.064 ± 0.003 | 0.065 ± 0.000 | 0.047 ± 0.003 | 0.040 ± 0.015 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.375 ± 0.006 | 0.158 ± 0.020 | 0.192 ± 0.006 | 0.061 ± 0.016 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 5.463 ± 0.184 | 4.527 ± 0.241 | 7.920 ± 1.884 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 26.640 ± 0.352 | 26.938 ± 0.512 | 34.606 ± 1.584 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 5.047 ± 0.034 | 5.212 ± 0.182 | 7.518 ± 2.060 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 23.537 ± 0.140 | 25.770 ± 0.251 | 48.184 ± 2.333 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.371 ± 0.007 | 0.091 ± 0.000 | 0.157 ± 0.028 | 0.064 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.745 ± 0.007 | 0.983 ± 0.019 | 1.299 ± 0.053 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 2.323 ± 0.039 | 2.501 ± 0.012 | 4.628 ± 0.258 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.707 ± 0.005 | 1.335 ± 0.060 | 1.260 ± 0.055 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 2.201 ± 0.112 | 4.250 ± 0.095 | 5.492 ± 0.227 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.187 ± 0.027 | 0.902 ± 0.004 | 0.970 ± 0.018 | 1.457 ± 0.316 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 13.817 ± 0.954 | 15.821 ± 0.251 | 19.151 ± 9.673 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 70.102 ± 1.438 | 87.144 ± 0.596 | 107.385 ± 10.565 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 13.496 ± 0.239 | 14.766 ± 0.111 | 18.080 ± 6.120 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 72.041 ± 5.018 | 79.937 ± 0.899 | 104.071 ± 7.663 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.171 ± 0.012 | 0.136 ± 0.000 | 0.156 ± 0.011 | 0.326 ± 0.099 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.694 ± 0.082 | 0.496 ± 0.001 | 0.484 ± 0.029 | 0.714 ± 0.079 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 2.074 ± 0.233 | 1.645 ± 0.016 | 2.458 ± 0.190 | 2.874 ± 0.392 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.280 ± 0.014 | 0.215 ± 0.001 | 0.207 ± 0.002 | 0.161 ± 0.008 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.036 ± 0.001 | 0.040 ± 0.000 | 0.050 ± 0.007 | 0.084 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.062 ± 0.003 | 0.067 ± 0.001 | 0.087 ± 0.001 | 0.143 ± 0.000 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.090 ± 0.001 | 0.091 ± 0.001 | 0.111 ± 0.001 | 0.255 ± 0.084 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.991 ± 0.033 | 0.780 ± 0.063 | 0.843 ± 0.024 | 0.834 ± 0.036 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.016 ± 0.001 | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.002 | 0.144 ± 0.013 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.176 ± 0.012 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.195 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.063 ± 0.000 | 0.198 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.066 ± 0.005 | 0.204 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.074 ± 0.002 | 0.211 ± 0.004 | 0.032 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.066 ± 0.000 | 0.198 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.071 ± 0.000 | 0.202 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.075 ± 0.000 | 0.205 ± 0.003 | 0.045 ± 0.016 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.064 ± 0.000 | 0.301 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.065 ± 0.001 | 0.296 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.066 ± 0.001 | 0.301 ± 0.005 | 0.057 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.066 ± 0.000 | 0.278 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.068 ± 0.001 | 0.276 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.069 ± 0.000 | 0.278 ± 0.003 | 0.054 ± 0.016 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.231 ± 0.003 | 0.034 ± 0.000 | 0.068 ± 0.010 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.280 ± 0.002 | 0.035 ± 0.000 | 0.075 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.276 ± 0.011 | 0.035 ± 0.000 | 0.075 ± 0.001 | 0.034 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.110 ± 0.000 | 0.227 ± 0.017 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.111 ± 0.000 | 0.229 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.103 ± 0.003 | 0.227 ± 0.004 | 0.052 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.111 ± 0.000 | 0.259 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.114 ± 0.001 | 0.264 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.116 ± 0.001 | 0.260 ± 0.009 | 0.041 ± 0.010 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.322 ± 0.003 | 0.049 ± 0.001 | 0.107 ± 0.019 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.370 ± 0.014 | 0.049 ± 0.004 | 0.124 ± 0.000 | 0.016 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.349 ± 0.019 | 0.051 ± 0.000 | 0.104 ± 0.008 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.036 ± 0.000 | 0.415 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.036 ± 0.001 | 0.421 ± 0.008 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.037 ± 0.000 | 0.415 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.037 ± 0.001 | 0.249 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.038 ± 0.000 | 0.250 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.038 ± 0.000 | 0.250 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.278 ± 0.002 | 0.037 ± 0.000 | 0.097 ± 0.013 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.323 ± 0.005 | 0.044 ± 0.001 | 0.117 ± 0.001 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.317 ± 0.002 | 0.047 ± 0.001 | 0.111 ± 0.008 | 0.042 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.038 ± 0.000 | 0.265 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.043 ± 0.005 | 0.271 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.055 ± 0.004 | 0.269 ± 0.031 | 0.041 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.038 ± 0.000 | 0.217 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.044 ± 0.000 | 0.221 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.050 ± 0.004 | 0.195 ± 0.003 | 0.044 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.020 ± 0.000 | 0.025 ± 0.000 | 0.016 ± 0.003 | 0.022 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.005 ± 0.001 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.041 ± 0.000 | 0.056 ± 0.000 | 0.061 ± 0.008 | 0.045 ± 0.031 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.002 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.009 ± 0.001 | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.003 | 0.009 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.027 ± 0.001 | 0.025 ± 0.003 | 0.025 ± 0.003 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 25.5x faster than the slowest successful cell (`tenferro-eager`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.6x faster than the slowest successful cell (`tenferro-eager`, 0.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 35.3x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 33.2x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 27.0x faster than the slowest successful cell (`jax-cpu`, 0.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.195 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.301 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 26.4x faster than the slowest successful cell (`tenferro-eager`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 26.1x faster than the slowest successful cell (`tenferro-eager`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 24.8x faster than the slowest successful cell (`tenferro-eager`, 0.349 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 32.2x faster than the slowest successful cell (`pytorch-cpu`, 0.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-eager`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 0.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

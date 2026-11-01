# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `4cc17b1c7061855cfd9ddc2d549a09f50199de3c`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_012518/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_012518/run.yaml`
- Timestamp: `2026-11-01T01:25:18.859596+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261101_012518`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_012518/cpu_ops_t1_20261101_012518.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.048 ± 0.000 | 0.051 ± 0.000 | 0.041 ± 0.001 | 0.032 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.000 | 0.044 ± 0.000 | 0.040 ± 0.001 | 0.044 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.147 ± 0.000 | 0.154 ± 0.001 | 0.134 ± 0.002 | 0.130 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.180 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.188 ± 0.035 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.148 ± 0.002 | 0.151 ± 0.000 | 0.176 ± 0.001 | 0.202 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.221 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.000 | 0.222 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.044 ± 0.000 | 0.039 ± 0.001 | 0.059 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.093 ± 0.001 | 0.094 ± 0.000 | 0.082 ± 0.001 | 0.098 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.084 ± 0.002 | 0.083 ± 0.000 | 0.076 ± 0.001 | 0.078 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.306 ± 0.001 | 0.308 ± 0.003 | 0.263 ± 0.003 | 0.256 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.146 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.150 ± 0.004 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.150 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.175 ± 0.004 | 0.051 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.136 ± 0.002 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.144 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.140 ± 0.002 | 0.040 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.000 | 0.153 ± 0.003 | 0.078 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.306 ± 0.006 | 0.302 ± 0.000 | 0.323 ± 0.001 | 0.392 ± 0.010 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 8.533 ± 0.075 | 10.284 ± 0.126 | 9.047 ± 0.021 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 55.960 ± 0.486 | 69.611 ± 0.072 | 56.722 ± 0.187 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 8.562 ± 0.077 | 8.493 ± 0.207 | 9.059 ± 0.089 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 54.542 ± 1.532 | 54.987 ± 0.207 | 58.355 ± 0.597 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 5.016 ± 0.065 | 12.048 ± 0.085 | 4.319 ± 0.048 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 33.742 ± 0.125 | 83.877 ± 1.070 | 32.452 ± 1.281 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 5.266 ± 0.020 | 5.375 ± 0.091 | 4.297 ± 0.046 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 35.327 ± 0.625 | 42.467 ± 0.338 | 33.271 ± 1.287 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.039 ± 0.001 | 0.026 ± 0.000 | 0.032 ± 0.000 | 0.034 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.002 | 0.147 ± 0.005 | 0.077 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 7.150 ± 0.093 | 7.462 ± 0.030 | 7.098 ± 0.055 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 51.731 ± 0.510 | 54.928 ± 0.386 | 49.395 ± 2.430 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 7.098 ± 0.209 | 7.447 ± 0.032 | 7.204 ± 0.039 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 48.009 ± 0.162 | 54.337 ± 0.406 | 53.311 ± 3.156 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.070 ± 0.002 | 0.174 ± 0.007 | 0.058 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.731 ± 0.003 | 1.247 ± 0.021 | 0.703 ± 0.011 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.118 ± 0.106 | 5.703 ± 0.057 | 4.078 ± 0.032 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.735 ± 0.004 | 1.728 ± 0.016 | 0.673 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.132 ± 0.059 | 9.819 ± 0.173 | 4.003 ± 0.027 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.685 ± 0.002 | 0.778 ± 0.008 | 0.661 ± 0.009 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 14.609 ± 0.064 | 18.735 ± 0.119 | 15.637 ± 0.211 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 101.813 ± 0.754 | 144.616 ± 0.906 | 107.506 ± 0.521 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 14.736 ± 0.175 | 15.896 ± 0.071 | 17.245 ± 0.164 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 101.446 ± 0.323 | 120.673 ± 0.806 | 117.131 ± 1.587 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.126 ± 0.003 | 0.127 ± 0.001 | 0.197 ± 0.004 | 0.200 ± 0.009 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.828 ± 0.007 | 0.831 ± 0.007 | 1.040 ± 0.004 | 1.229 ± 0.072 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.297 ± 0.100 | 3.289 ± 0.080 | 3.560 ± 0.102 | 4.651 ± 0.185 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.095 ± 0.000 | 0.094 ± 0.001 | 0.113 ± 0.003 | 0.106 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.034 ± 0.001 | 0.038 ± 0.000 | 0.050 ± 0.001 | 0.047 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.046 ± 0.002 | 0.050 ± 0.003 | 0.066 ± 0.001 | 0.076 ± 0.012 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.080 ± 0.000 | 0.086 ± 0.000 | 0.120 ± 0.002 | 0.160 ± 0.002 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.579 ± 0.010 | 0.609 ± 0.006 | 0.643 ± 0.011 | 0.594 ± 0.008 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.175 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.178 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.191 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.034 ± 0.001 | 0.197 ± 0.006 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.040 ± 0.000 | 0.202 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.044 ± 0.000 | 0.207 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.036 ± 0.000 | 0.194 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.042 ± 0.001 | 0.198 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.046 ± 0.000 | 0.205 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.031 ± 0.001 | 0.277 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.032 ± 0.000 | 0.274 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.000 | 0.279 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.030 ± 0.000 | 0.255 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.031 ± 0.001 | 0.256 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.033 ± 0.001 | 0.262 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.072 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.074 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.074 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.057 ± 0.000 | 0.221 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.059 ± 0.000 | 0.220 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.062 ± 0.000 | 0.225 ± 0.009 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.061 ± 0.001 | 0.246 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.063 ± 0.000 | 0.248 ± 0.009 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.066 ± 0.000 | 0.251 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.121 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.124 ± 0.002 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.125 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.021 ± 0.000 | 0.411 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.021 ± 0.000 | 0.416 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.022 ± 0.000 | 0.417 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.022 ± 0.000 | 0.246 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.021 ± 0.000 | 0.248 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.023 ± 0.000 | 0.248 ± 0.002 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.108 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.001 | 0.116 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.127 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.021 ± 0.000 | 0.263 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.027 ± 0.000 | 0.271 ± 0.003 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.038 ± 0.000 | 0.285 ± 0.004 | 0.019 ± 0.017 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.001 | 0.213 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.027 ± 0.000 | 0.219 ± 0.002 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.039 ± 0.000 | 0.230 ± 0.003 | 0.036 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.040 ± 0.001 | 0.041 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 36.2x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 28.3x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 40.2x faster than the slowest successful cell (`jax-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 32.2x faster than the slowest successful cell (`jax-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 53.0x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 51.0x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 53.3x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 20.8x faster than the slowest successful cell (`pytorch-cpu`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.251 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 33.0x faster than the slowest successful cell (`pytorch-cpu`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 30.2x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_020627/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_020627/run.yaml`
- Timestamp: `2026-11-01T02:06:27.548364+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261101_020627`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_020627/cpu_ops_t4_20261101_020627.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.049 ± 0.000 | 0.049 ± 0.000 | 0.041 ± 0.001 | 0.047 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.044 ± 0.001 | 0.043 ± 0.000 | 0.040 ± 0.001 | 0.044 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.150 ± 0.002 | 0.150 ± 0.000 | 0.132 ± 0.002 | 0.168 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.179 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.230 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.265 ± 0.000 | 0.268 ± 0.002 | 0.125 ± 0.000 | 0.335 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.193 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.000 | 0.237 ± 0.051 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.031 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.000 | 0.046 ± 0.000 | 0.059 ± 0.001 | 0.065 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.037 ± 0.000 | 0.038 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.033 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.040 ± 0.000 | 0.042 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.049 ± 0.000 | 0.049 ± 0.000 | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.092 ± 0.000 | 0.092 ± 0.001 | 0.078 ± 0.002 | 0.096 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.085 ± 0.000 | 0.084 ± 0.001 | 0.075 ± 0.001 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.313 ± 0.007 | 0.307 ± 0.002 | 0.262 ± 0.003 | 0.258 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.145 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.149 ± 0.001 | 0.016 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.151 ± 0.004 | 0.021 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.001 | 0.176 ± 0.003 | 0.060 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.002 | 0.179 ± 0.002 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.064 ± 0.001 | 0.190 ± 0.002 | 0.055 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.000 | 0.183 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.071 ± 0.001 | 0.201 ± 0.002 | 0.095 ± 0.005 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.350 ± 0.000 | 0.351 ± 0.006 | 0.344 ± 0.006 | 0.521 ± 0.011 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 34.999 ± 0.998 | 7.194 ± 0.160 | 11.395 ± 0.333 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 47.661 ± 0.823 | 42.654 ± 0.855 | 49.418 ± 2.325 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 25.993 ± 0.022 | 5.982 ± 0.172 | 11.437 ± 0.172 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 46.892 ± 0.700 | 31.114 ± 1.008 | 48.980 ± 1.653 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 45.969 ± 6.001 | 7.567 ± 0.134 | 6.078 ± 0.873 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 50.150 ± 15.182 | 47.712 ± 0.554 | 32.943 ± 4.462 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 37.021 ± 3.769 | 3.733 ± 0.082 | 4.499 ± 4.076 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 40.422 ± 25.912 | 24.319 ± 0.228 | 35.428 ± 4.078 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.063 ± 0.003 | 0.039 ± 0.000 | 0.035 ± 0.001 | 0.039 ± 0.009 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.104 ± 0.001 | 0.177 ± 0.005 | 0.111 ± 0.007 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 35.973 ± 16.589 | 5.107 ± 0.292 | 8.989 ± 1.836 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 70.582 ± 23.069 | 33.982 ± 0.169 | 48.321 ± 11.157 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 57.009 ± 8.231 | 5.338 ± 0.031 | 10.278 ± 1.919 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 57.028 ± 52.437 | 32.681 ± 0.249 | 54.103 ± 9.679 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.123 ± 0.001 | 0.174 ± 0.002 | 0.073 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 2.997 ± 0.027 | 2.993 ± 1.917 | 0.921 ± 0.040 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 4.011 ± 0.894 | 3.012 ± 0.040 | 3.796 ± 0.047 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 2.999 ± 0.004 | 1.286 ± 0.052 | 0.961 ± 0.054 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 4.081 ± 4.919 | 4.832 ± 0.075 | 3.749 ± 0.023 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.928 ± 0.004 | 0.995 ± 0.014 | 0.900 ± 0.166 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 21.689 ± 0.179 | 16.548 ± 0.081 | 16.439 ± 1.233 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 90.717 ± 11.655 | 92.877 ± 1.104 | 101.943 ± 7.093 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 32.038 ± 0.169 | 14.783 ± 0.163 | 17.681 ± 0.845 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 103.968 ± 6.761 | 82.403 ± 0.513 | 105.902 ± 13.885 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.236 ± 0.005 | 0.104 ± 0.001 | 0.160 ± 0.012 | 0.189 ± 0.054 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.876 ± 0.004 | 0.519 ± 0.011 | 0.682 ± 0.049 | 0.867 ± 0.209 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 3.059 ± 0.023 | 1.889 ± 0.022 | 2.453 ± 0.158 | 3.315 ± 0.197 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.171 ± 0.003 | 0.181 ± 0.000 | 0.177 ± 0.003 | 0.112 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.056 ± 0.000 | 0.065 ± 0.000 | 0.050 ± 0.001 | 0.085 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.071 ± 0.003 | 0.083 ± 0.003 | 0.097 ± 0.006 | 0.154 ± 0.024 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.092 ± 0.011 | 0.105 ± 0.002 | 0.137 ± 0.019 | 0.196 ± 0.002 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 1.860 ± 1.915 | 0.824 ± 0.002 | 0.834 ± 0.004 | 0.842 ± 0.017 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.000 | 0.178 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.185 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.190 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.050 ± 0.001 | 0.196 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.056 ± 0.000 | 0.201 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.061 ± 0.001 | 0.211 ± 0.006 | 0.019 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.051 ± 0.001 | 0.194 ± 0.005 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.057 ± 0.000 | 0.200 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.062 ± 0.000 | 0.208 ± 0.003 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.052 ± 0.002 | 0.383 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.052 ± 0.001 | 0.381 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.054 ± 0.001 | 0.385 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.049 ± 0.000 | 0.336 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.050 ± 0.001 | 0.336 ± 0.011 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.051 ± 0.000 | 0.342 ± 0.008 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.071 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.073 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.074 ± 0.001 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.085 ± 0.001 | 0.284 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.087 ± 0.000 | 0.286 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.090 ± 0.000 | 0.294 ± 0.004 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.087 ± 0.002 | 0.317 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.088 ± 0.001 | 0.319 ± 0.009 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.091 ± 0.002 | 0.325 ± 0.009 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.001 | 0.117 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.001 | 0.121 ± 0.003 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.123 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.033 ± 0.000 | 0.409 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.033 ± 0.000 | 0.411 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.034 ± 0.001 | 0.417 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.037 ± 0.003 | 0.246 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.034 ± 0.000 | 0.246 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.035 ± 0.000 | 0.250 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.105 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.114 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.001 | 0.127 ± 0.002 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.031 ± 0.000 | 0.263 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.036 ± 0.001 | 0.270 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.048 ± 0.000 | 0.289 ± 0.013 | 0.032 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.030 ± 0.000 | 0.216 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.036 ± 0.001 | 0.222 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.048 ± 0.001 | 0.236 ± 0.002 | 0.026 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.014 ± 0.000 | 0.017 ± 0.003 | 0.028 ± 11.122 | 0.013 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.029 ± 0.000 | 0.031 ± 0.000 | 0.036 ± 0.001 | 0.041 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.001 | 0.012 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.009 ± 0.000 | 0.011 ± 0.005 | 0.011 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.023 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`jax-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 57.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 39.3x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 39.0x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 29.3x faster than the slowest successful cell (`pytorch-cpu`, 0.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.7x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.294 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.8x faster than the slowest successful cell (`pytorch-cpu`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.411 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 28.7x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

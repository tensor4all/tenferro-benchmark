# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`
- Benchmark commit: `aecee8b998098c968f4a0f59d6228ac81d3e7b09`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261021_180521/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261021_180521`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261021_180521`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261021_180521/cpu_ops_t1_20261021_180521.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261021_180521/cpu_ops_t1_20261021_180521.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.048 ± 0.000 | 0.049 ± 0.000 | 0.041 ± 0.001 | 0.033 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.042 ± 0.000 | 0.043 ± 0.000 | 0.040 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.148 ± 0.002 | 0.149 ± 0.000 | 0.135 ± 0.001 | 0.129 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.179 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.189 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.148 ± 0.001 | 0.153 ± 0.000 | 0.176 ± 0.002 | 0.202 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.226 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.000 | 0.190 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.021 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.045 ± 0.000 | 0.039 ± 0.000 | 0.059 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.027 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.093 ± 0.000 | 0.094 ± 0.000 | 0.084 ± 0.000 | 0.095 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.083 ± 0.000 | 0.083 ± 0.000 | 0.076 ± 0.001 | 0.076 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.306 ± 0.001 | 0.307 ± 0.000 | 0.264 ± 0.002 | 0.250 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.146 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.154 ± 0.002 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.001 | 0.152 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.177 ± 0.003 | 0.036 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.136 ± 0.002 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.049 ± 0.000 | 0.148 ± 0.002 | 0.041 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.146 ± 0.002 | 0.023 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.056 ± 0.001 | 0.159 ± 0.006 | 0.057 ± 0.022 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.304 ± 0.001 | 0.305 ± 0.024 | 0.324 ± 0.002 | 0.392 ± 0.009 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 8.491 ± 0.794 | 10.274 ± 0.069 | 9.117 ± 0.018 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 56.130 ± 0.235 | 69.146 ± 0.213 | 61.631 ± 0.523 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 8.443 ± 0.063 | 8.487 ± 0.108 | 9.147 ± 0.057 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 56.302 ± 0.309 | 53.789 ± 0.347 | 61.076 ± 0.329 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 5.011 ± 0.020 | 12.573 ± 0.230 | 4.465 ± 0.024 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 34.626 ± 0.202 | 84.166 ± 0.964 | 33.900 ± 0.607 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 5.353 ± 0.043 | 5.389 ± 0.040 | 4.698 ± 0.086 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 35.645 ± 0.229 | 44.399 ± 0.252 | 34.690 ± 1.350 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.040 ± 0.000 | 0.027 ± 0.000 | 0.033 ± 0.001 | 0.035 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.078 ± 0.001 | 0.150 ± 0.036 | 0.077 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 7.332 ± 0.343 | 7.323 ± 0.043 | 7.246 ± 0.063 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 54.270 ± 0.310 | 58.555 ± 0.824 | 50.721 ± 0.246 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 7.155 ± 0.146 | 7.370 ± 0.074 | 7.593 ± 0.194 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 51.128 ± 0.593 | 57.079 ± 0.492 | 56.344 ± 0.578 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.001 | 0.180 ± 0.005 | 0.059 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.744 ± 0.007 | 1.257 ± 0.009 | 0.670 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.185 ± 0.040 | 5.805 ± 0.020 | 4.167 ± 0.032 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.745 ± 0.012 | 1.750 ± 0.018 | 0.680 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 4.146 ± 0.047 | 9.836 ± 0.020 | 4.045 ± 0.035 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.666 ± 0.003 | 0.789 ± 0.004 | 0.642 ± 0.016 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 14.781 ± 0.090 | 21.236 ± 0.111 | 15.598 ± 0.066 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 104.889 ± 0.197 | 146.257 ± 0.881 | 114.916 ± 0.331 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 14.801 ± 0.100 | 16.088 ± 0.051 | 15.744 ± 0.059 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 104.056 ± 1.787 | 121.195 ± 0.428 | 117.491 ± 1.372 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.126 ± 0.001 | 0.128 ± 0.001 | 0.196 ± 0.004 | 0.178 ± 0.007 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.853 ± 0.008 | 0.884 ± 0.031 | 1.072 ± 0.019 | 1.338 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.342 ± 0.066 | 3.370 ± 0.035 | 3.549 ± 0.104 | 4.917 ± 0.197 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.095 ± 0.002 | 0.095 ± 0.001 | 0.114 ± 0.003 | 0.109 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.034 ± 0.001 | 0.037 ± 0.000 | 0.050 ± 0.001 | 0.047 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.046 ± 0.002 | 0.053 ± 0.017 | 0.067 ± 0.004 | 0.085 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.081 ± 0.005 | 0.087 ± 0.001 | 0.122 ± 0.015 | 0.170 ± 0.008 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.589 ± 0.010 | 0.581 ± 0.003 | 0.637 ± 0.002 | 0.597 ± 0.007 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.179 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.179 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.194 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.038 ± 0.000 | 0.198 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.044 ± 0.001 | 0.201 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.049 ± 0.001 | 0.208 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.038 ± 0.000 | 0.193 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.045 ± 0.000 | 0.199 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.049 ± 0.000 | 0.203 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.034 ± 0.000 | 0.279 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.034 ± 0.000 | 0.273 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.036 ± 0.000 | 0.276 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.034 ± 0.000 | 0.261 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.034 ± 0.001 | 0.260 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.037 ± 0.000 | 0.261 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.073 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.075 ± 0.001 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.075 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.064 ± 0.000 | 0.220 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.066 ± 0.001 | 0.218 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.068 ± 0.001 | 0.221 ± 0.007 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.066 ± 0.001 | 0.248 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.067 ± 0.000 | 0.252 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.070 ± 0.000 | 0.253 ± 0.007 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.001 | 0.122 ± 0.002 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.124 ± 0.002 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.125 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.023 ± 0.000 | 0.416 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.023 ± 0.000 | 0.427 ± 0.021 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.024 ± 0.000 | 0.420 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.023 ± 0.011 | 0.244 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.024 ± 0.000 | 0.248 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.025 ± 0.000 | 0.248 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.110 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.116 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.131 ± 0.036 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.023 ± 0.000 | 0.268 ± 0.005 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.028 ± 0.000 | 0.271 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.040 ± 0.000 | 0.284 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.026 ± 0.000 | 0.214 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.032 ± 0.000 | 0.219 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.043 ± 0.000 | 0.227 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.001 | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.033 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 34.8x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 24.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 41.1x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 29.7x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.4x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 53.8x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 53.1x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 51.9x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.279 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.7x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 32.5x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 32.9x faster than the slowest successful cell (`pytorch-cpu`, 0.427 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.427 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 29.4x faster than the slowest successful cell (`pytorch-cpu`, 0.420 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.420 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261021_183748/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261021_183748`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261021_183748`.

- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

### CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261021_183748/cpu_ops_t4_20261021_183748.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261021_183748/cpu_ops_t4_20261021_183748.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.048 ± 0.000 | 0.041 ± 0.001 | 0.056 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.001 | 0.043 ± 0.000 | 0.039 ± 0.001 | 0.044 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.146 ± 0.001 | 0.147 ± 0.001 | 0.132 ± 0.001 | 0.133 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.178 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | 0.187 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.265 ± 0.000 | 0.269 ± 0.000 | 0.126 ± 0.001 | 0.339 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.188 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.189 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.032 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.045 ± 0.000 | 0.060 ± 0.001 | 0.079 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.031 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.037 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.033 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.040 ± 0.000 | 0.053 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.047 ± 0.000 | 0.028 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.000 | 0.090 ± 0.000 | 0.078 ± 0.002 | 0.094 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.081 ± 0.000 | 0.083 ± 0.001 | 0.075 ± 0.000 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.302 ± 0.004 | 0.306 ± 0.002 | 0.262 ± 0.001 | 0.255 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.144 ± 0.002 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.148 ± 0.001 | 0.016 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.150 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.000 | 0.174 ± 0.001 | 0.090 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.000 | 0.176 ± 0.004 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.049 ± 0.001 | 0.192 ± 0.006 | 0.068 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.182 ± 0.002 | 0.030 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.056 ± 0.001 | 0.200 ± 0.003 | 0.081 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.345 ± 0.001 | 0.345 ± 0.004 | 0.343 ± 0.006 | 0.520 ± 0.034 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 6.721 ± 0.073 | 7.231 ± 0.156 | 10.435 ± 0.594 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 33.986 ± 0.383 | 42.234 ± 0.806 | 55.924 ± 5.132 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 6.616 ± 0.057 | 5.989 ± 0.062 | 10.902 ± 0.486 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 34.134 ± 0.287 | 32.157 ± 0.627 | 61.361 ± 8.743 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 3.319 ± 0.044 | 7.757 ± 0.217 | 5.619 ± 1.108 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 19.552 ± 0.192 | 46.842 ± 0.521 | 29.995 ± 7.969 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 3.843 ± 0.066 | 3.711 ± 0.039 | 4.380 ± 3.505 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 20.430 ± 0.075 | 24.355 ± 0.300 | 37.458 ± 6.032 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.056 ± 0.002 | 0.035 ± 0.000 | 0.036 ± 0.002 | 0.038 ± 0.005 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.102 ± 0.001 | 0.180 ± 0.004 | 0.102 ± 0.022 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 5.394 ± 0.047 | 5.091 ± 0.070 | 10.082 ± 0.728 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 31.240 ± 0.164 | 33.490 ± 0.347 | 58.586 ± 5.405 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 5.303 ± 0.076 | 5.473 ± 0.088 | 10.486 ± 0.722 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 28.440 ± 2.971 | 32.962 ± 0.414 | 50.511 ± 5.380 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.106 ± 0.001 | 0.172 ± 0.008 | 0.101 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.582 ± 0.007 | 4.887 ± 17.879 | 0.912 ± 0.134 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 2.360 ± 0.040 | 3.119 ± 0.150 | 3.814 ± 0.386 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.603 ± 0.005 | 1.271 ± 0.011 | 0.909 ± 0.050 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 2.421 ± 0.061 | 4.960 ± 0.132 | 3.733 ± 0.050 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.914 ± 0.008 | 1.005 ± 0.014 | 0.889 ± 0.011 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 13.068 ± 0.135 | 16.400 ± 0.102 | 17.583 ± 1.431 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 80.475 ± 0.400 | 91.003 ± 1.022 | 110.285 ± 5.109 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 13.137 ± 0.069 | 14.742 ± 0.091 | 17.137 ± 0.692 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 80.063 ± 0.483 | 80.849 ± 0.760 | 111.744 ± 3.953 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.099 ± 0.001 | 0.098 ± 0.001 | 0.154 ± 0.010 | 0.180 ± 0.031 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.496 ± 0.004 | 0.498 ± 0.007 | 0.720 ± 0.034 | 0.746 ± 0.070 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.861 ± 0.032 | 1.877 ± 0.015 | 2.598 ± 0.268 | 3.023 ± 0.218 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.176 ± 0.001 | 0.175 ± 0.000 | 0.179 ± 0.004 | 0.115 ± 0.010 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.053 ± 0.001 | 0.060 ± 0.001 | 0.051 ± 0.001 | 0.073 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.068 ± 0.003 | 0.079 ± 0.001 | 0.096 ± 0.005 | 0.152 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.088 ± 0.001 | 0.100 ± 0.009 | 0.139 ± 0.019 | 0.196 ± 0.028 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.817 ± 0.004 | 0.813 ± 0.005 | 0.830 ± 0.007 | 0.846 ± 0.011 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.177 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.177 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.001 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.231 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.039 ± 0.000 | 0.197 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.045 ± 0.000 | 0.201 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.050 ± 0.001 | 0.209 ± 0.005 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.039 ± 0.001 | 0.193 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.046 ± 0.004 | 0.201 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.061 ± 0.028 | 0.203 ± 0.004 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.036 ± 0.000 | 0.385 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.036 ± 0.000 | 0.379 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.037 ± 0.000 | 0.381 ± 0.003 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.037 ± 0.000 | 0.336 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.038 ± 0.000 | 0.332 ± 0.009 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.039 ± 0.000 | 0.337 ± 0.012 | 0.021 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.006 | 0.073 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.075 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.074 ± 0.001 | 0.016 ± 0.004 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.067 ± 0.001 | 0.289 ± 0.005 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.068 ± 0.000 | 0.286 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.070 ± 0.001 | 0.288 ± 0.007 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.068 ± 0.000 | 0.315 ± 0.009 | 0.013 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.070 ± 0.001 | 0.316 ± 0.007 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.079 ± 0.026 | 0.317 ± 0.009 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.119 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.122 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.122 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.023 ± 0.000 | 0.410 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.023 ± 0.000 | 0.406 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.024 ± 0.000 | 0.412 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.024 ± 0.001 | 0.243 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.023 ± 0.000 | 0.246 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.024 ± 0.001 | 0.248 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.106 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.001 | 0.115 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.044 ± 0.001 | 0.126 ± 0.001 | 0.034 ± 0.006 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.024 ± 0.000 | 0.263 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.029 ± 0.000 | 0.270 ± 0.004 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.041 ± 0.000 | 0.284 ± 0.005 | 0.038 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.027 ± 0.000 | 0.212 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.033 ± 0.000 | 0.219 ± 0.008 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.044 ± 0.000 | 0.231 ± 0.005 | 0.026 ± 0.011 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.012 ± 0.001 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.026 ± 0.001 | 0.030 ± 0.001 | 0.037 ± 0.001 | 0.038 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.049 ± 0.035 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.021 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.028 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 34.3x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 33.4x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 51.5x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 57.8x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 28.6x faster than the slowest successful cell (`pytorch-cpu`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.379 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.379 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 0.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.9x faster than the slowest successful cell (`pytorch-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.8x faster than the slowest successful cell (`pytorch-cpu`, 0.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.5x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.2x faster than the slowest successful cell (`pytorch-cpu`, 0.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

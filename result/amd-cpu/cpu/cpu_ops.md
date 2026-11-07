# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `d6655b567e09aa29c3201655e1de418716934876`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261107_180711/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_180711/run.yaml`
- Timestamp: `2026-11-07T18:07:11.532995+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261107_180711`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261107_180711/cpu_ops_t1_20261107_180711.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.046 ± 0.000 | 0.040 ± 0.001 | 0.036 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.043 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.147 ± 0.000 | 0.146 ± 0.001 | 0.132 ± 0.001 | 0.128 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.178 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.188 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.163 ± 0.001 | 0.164 ± 0.000 | 0.174 ± 0.002 | 0.198 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.191 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.001 | 0.189 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.029 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.042 ± 0.000 | 0.039 ± 0.001 | 0.058 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.024 ± 0.000 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.027 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.000 | 0.034 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.093 ± 0.000 | 0.091 ± 0.000 | 0.081 ± 0.001 | 0.094 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.081 ± 0.001 | 0.079 ± 0.000 | 0.076 ± 0.001 | 0.076 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.302 ± 0.000 | 0.302 ± 0.003 | 0.263 ± 0.000 | 0.248 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.143 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.147 ± 0.002 | 0.013 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.150 ± 0.002 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.174 ± 0.001 | 0.033 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.048 ± 0.000 | 0.134 ± 0.002 | 0.023 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.075 ± 0.000 | 0.143 ± 0.001 | 0.039 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.000 | 0.137 ± 0.002 | 0.023 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.089 ± 0.000 | 0.150 ± 0.001 | 0.054 ± 0.002 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.298 ± 0.002 | 0.296 ± 0.000 | 0.322 ± 0.001 | 0.384 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 16.750 ± 0.070 | 11.487 ± 0.045 | 10.209 ± 0.055 | 8.909 ± 0.033 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 106.060 ± 0.529 | 74.613 ± 0.267 | 65.218 ± 0.110 | 57.853 ± 0.312 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.297 ± 0.092 | 11.623 ± 0.050 | 8.451 ± 0.075 | 8.881 ± 0.038 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 60.321 ± 0.391 | 74.750 ± 0.576 | 51.159 ± 0.240 | 57.019 ± 0.170 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 10.942 ± 0.056 | 5.631 ± 0.048 | 12.388 ± 0.057 | 4.349 ± 0.059 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 74.338 ± 1.201 | 37.988 ± 0.304 | 78.165 ± 1.136 | 30.928 ± 0.184 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 10.052 ± 0.032 | 5.889 ± 0.027 | 5.401 ± 0.039 | 4.298 ± 0.017 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 71.749 ± 0.928 | 39.254 ± 0.681 | 41.697 ± 0.671 | 31.692 ± 0.112 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.033 ± 0.001 | 0.025 ± 0.000 | 0.028 ± 0.001 | 0.031 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.000 | 0.140 ± 0.001 | 0.085 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 16.481 ± 0.099 | 10.069 ± 1.653 | 7.274 ± 0.040 | 7.042 ± 0.028 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 114.530 ± 1.001 | 72.332 ± 0.550 | 50.679 ± 0.424 | 45.580 ± 2.127 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 12.514 ± 0.009 | 9.880 ± 0.064 | 7.299 ± 0.065 | 7.196 ± 0.035 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 90.044 ± 0.905 | 68.570 ± 0.540 | 50.366 ± 0.329 | 51.047 ± 2.264 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.146 ± 0.002 | 0.170 ± 0.001 | 0.053 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.822 ± 0.019 | 1.391 ± 0.007 | 1.240 ± 0.021 | 0.668 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 8.994 ± 0.116 | 8.018 ± 0.015 | 5.609 ± 0.098 | 4.140 ± 0.075 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.865 ± 0.035 | 1.389 ± 0.012 | 2.115 ± 0.032 | 0.659 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.427 ± 0.068 | 7.996 ± 0.044 | 9.597 ± 0.064 | 4.055 ± 0.051 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.856 ± 0.001 | 0.772 ± 0.004 | 0.647 ± 0.016 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 28.586 ± 0.388 | 19.612 ± 0.050 | 21.114 ± 0.113 | 15.379 ± 0.084 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 191.774 ± 3.339 | 139.758 ± 0.500 | 135.345 ± 0.744 | 109.819 ± 0.161 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.833 ± 0.043 | 19.655 ± 0.104 | 15.832 ± 0.035 | 15.371 ± 0.047 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 101.067 ± 0.239 | 140.888 ± 0.961 | 106.974 ± 0.448 | 115.285 ± 0.365 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.125 ± 0.003 | 0.125 ± 0.001 | 0.178 ± 0.005 | 0.178 ± 0.003 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.836 ± 0.007 | 0.826 ± 0.004 | 1.015 ± 0.020 | 1.251 ± 0.047 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.281 ± 0.051 | 3.297 ± 0.020 | 3.456 ± 0.023 | 4.820 ± 0.179 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.091 ± 0.000 | 0.090 ± 0.000 | 0.110 ± 0.001 | 0.101 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.034 ± 0.000 | 0.037 ± 0.000 | 0.049 ± 0.001 | 0.042 ± 0.000 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.046 ± 0.001 | 0.049 ± 0.003 | 0.069 ± 0.003 | 0.068 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.081 ± 0.000 | 0.086 ± 0.005 | 0.120 ± 0.001 | 0.145 ± 0.022 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.577 ± 0.019 | 0.563 ± 0.003 | 0.634 ± 0.003 | 0.582 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.171 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.175 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.190 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.251 ± 0.001 | 0.037 ± 0.001 | 0.196 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.266 ± 0.002 | 0.044 ± 0.001 | 0.199 ± 0.005 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.275 ± 0.003 | 0.051 ± 0.001 | 0.207 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.101 ± 0.001 | 0.039 ± 0.000 | 0.194 ± 0.006 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.110 ± 0.003 | 0.047 ± 0.000 | 0.199 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.115 ± 0.002 | 0.053 ± 0.000 | 0.206 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.692 ± 0.009 | 0.035 ± 0.000 | 0.277 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.702 ± 0.002 | 0.036 ± 0.000 | 0.275 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.717 ± 0.017 | 0.037 ± 0.000 | 0.276 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.192 ± 0.001 | 0.034 ± 0.000 | 0.256 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.194 ± 0.001 | 0.035 ± 0.000 | 0.258 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.198 ± 0.001 | 0.036 ± 0.000 | 0.258 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.071 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.072 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.074 ± 0.002 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.977 ± 0.025 | 0.061 ± 0.000 | 0.218 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.952 ± 0.003 | 0.064 ± 0.000 | 0.217 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.971 ± 0.004 | 0.066 ± 0.000 | 0.219 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.183 ± 0.000 | 0.065 ± 0.001 | 0.249 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.185 ± 0.002 | 0.067 ± 0.000 | 0.247 ± 0.006 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.191 ± 0.002 | 0.070 ± 0.000 | 0.249 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.119 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.001 | 0.123 ± 0.003 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.122 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.190 ± 0.001 | 0.026 ± 0.000 | 0.414 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.189 ± 0.002 | 0.027 ± 0.001 | 0.412 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.194 ± 0.003 | 0.028 ± 0.000 | 0.416 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.098 ± 0.001 | 0.026 ± 0.000 | 0.247 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.098 ± 0.000 | 0.027 ± 0.000 | 0.247 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.100 ± 0.002 | 0.028 ± 0.000 | 0.246 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.108 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.001 | 0.115 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.126 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.226 ± 0.002 | 0.024 ± 0.000 | 0.262 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.242 ± 0.002 | 0.033 ± 0.000 | 0.269 ± 0.006 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.264 ± 0.001 | 0.046 ± 0.001 | 0.283 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.094 ± 0.004 | 0.024 ± 0.000 | 0.214 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.102 ± 0.001 | 0.034 ± 0.000 | 0.219 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.113 ± 0.001 | 0.048 ± 0.001 | 0.230 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.037 ± 0.001 | 0.047 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 33.8x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 31.3x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.5x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 50.6x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 51.2x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 22.6x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 53.5x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.0x faster than the slowest successful cell (`tenferro-eager`, 0.251 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 54.7x faster than the slowest successful cell (`tenferro-eager`, 0.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 53.0x faster than the slowest successful cell (`tenferro-eager`, 0.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 49.8x faster than the slowest successful cell (`tenferro-eager`, 0.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 77.1x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 66.5x faster than the slowest successful cell (`tenferro-eager`, 0.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 56.2x faster than the slowest successful cell (`tenferro-eager`, 0.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.971 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 33.3x faster than the slowest successful cell (`pytorch-cpu`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.414 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 31.6x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 30.4x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.230 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261107_184743/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261107_184743/run.yaml`
- Timestamp: `2026-11-07T18:47:43.753266+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261107_184743`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261107_184743/cpu_ops_t4_20261107_184743.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.045 ± 0.000 | 0.039 ± 0.000 | 0.053 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.043 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.146 ± 0.000 | 0.144 ± 0.000 | 0.131 ± 0.000 | 0.131 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.182 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.189 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.166 ± 0.001 | 0.167 ± 0.001 | 0.121 ± 0.005 | 0.241 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.193 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.232 ± 0.045 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.031 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.023 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.042 ± 0.000 | 0.059 ± 0.001 | 0.063 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.000 | 0.015 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.036 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.032 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.039 ± 0.000 | 0.040 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.034 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.087 ± 0.001 | 0.077 ± 0.001 | 0.094 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.081 ± 0.001 | 0.080 ± 0.000 | 0.074 ± 0.001 | 0.076 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.302 ± 0.005 | 0.299 ± 0.000 | 0.261 ± 0.002 | 0.280 ± 0.030 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.145 ± 0.002 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.147 ± 0.003 | 0.012 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.150 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.046 ± 0.001 | 0.173 ± 0.001 | 0.079 ± 0.024 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.059 ± 0.000 | 0.178 ± 0.002 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.086 ± 0.000 | 0.188 ± 0.002 | 0.053 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.001 | 0.182 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.002 | 0.200 ± 0.001 | 0.062 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.526 ± 0.006 | 0.519 ± 0.003 | 0.333 ± 0.001 | 0.428 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 29.795 ± 0.130 | 26.382 ± 0.021 | 7.036 ± 0.278 | 11.207 ± 0.362 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 153.264 ± 0.180 | 117.613 ± 0.393 | 42.313 ± 0.975 | 49.628 ± 4.471 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 16.071 ± 0.066 | 26.326 ± 0.027 | 5.950 ± 0.022 | 11.230 ± 1.847 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 83.118 ± 0.543 | 116.655 ± 0.262 | 30.619 ± 1.416 | 56.790 ± 2.777 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 16.026 ± 0.067 | 9.281 ± 0.013 | 8.924 ± 0.123 | 5.989 ± 1.044 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 308.599 ± 50.791 | 301.506 ± 25.584 | 48.845 ± 1.114 | 34.737 ± 2.270 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 14.154 ± 0.041 | 9.793 ± 0.034 | 3.597 ± 0.021 | 4.211 ± 1.728 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 212.760 ± 4.408 | 303.113 ± 11.622 | 23.987 ± 0.295 | 28.543 ± 1.964 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.067 ± 0.000 | 0.055 ± 0.000 | 0.035 ± 0.001 | 0.038 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.159 ± 0.004 | 0.172 ± 0.001 | 0.074 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 169.000 ± 11.984 | 15.471 ± 11.404 | 4.785 ± 0.066 | 8.311 ± 1.995 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 440.201 ± 11.231 | 392.102 ± 9.753 | 33.408 ± 0.153 | 45.500 ± 10.112 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 17.175 ± 0.060 | 15.254 ± 11.360 | 5.304 ± 0.092 | 9.942 ± 1.182 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 256.507 ± 11.967 | 390.795 ± 11.919 | 33.507 ± 0.484 | 54.352 ± 10.593 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.156 ± 0.001 | 0.172 ± 0.001 | 0.063 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 3.051 ± 0.029 | 2.612 ± 0.028 | 2.463 ± 1.455 | 0.882 ± 0.060 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 290.403 ± 18.028 | 278.160 ± 26.905 | 2.959 ± 0.081 | 3.819 ± 0.120 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.488 ± 0.017 | 2.600 ± 0.029 | 1.401 ± 0.033 | 0.952 ± 0.119 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 136.578 ± 5.699 | 290.004 ± 15.973 | 4.762 ± 0.049 | 3.676 ± 0.045 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.500 ± 0.008 | 0.995 ± 0.006 | 0.899 ± 0.024 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 35.025 ± 0.562 | 24.657 ± 0.081 | 16.322 ± 0.030 | 19.113 ± 0.984 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 234.185 ± 2.850 | 176.835 ± 0.349 | 93.543 ± 1.681 | 107.864 ± 4.411 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 20.009 ± 0.138 | 26.228 ± 0.086 | 14.719 ± 0.035 | 16.447 ± 1.436 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 124.764 ± 0.936 | 177.059 ± 0.170 | 82.934 ± 0.419 | 111.894 ± 7.860 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.171 ± 0.001 | 0.172 ± 0.001 | 0.140 ± 0.004 | 0.165 ± 0.038 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.913 ± 0.005 | 0.913 ± 0.004 | 0.635 ± 0.015 | 0.766 ± 0.075 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 3.598 ± 0.016 | 3.592 ± 0.028 | 2.375 ± 0.179 | 2.618 ± 0.119 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.280 ± 0.003 | 0.277 ± 0.001 | 0.177 ± 0.001 | 0.105 ± 0.016 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.035 ± 0.000 | 0.040 ± 0.000 | 0.048 ± 0.000 | 0.080 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.075 ± 0.002 | 0.082 ± 0.002 | 0.094 ± 0.005 | 0.152 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.111 ± 0.006 | 0.118 ± 0.004 | 0.134 ± 0.001 | 0.194 ± 0.017 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 1.093 ± 0.031 | 1.093 ± 0.003 | 0.829 ± 0.010 | 0.837 ± 0.011 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.003 | 0.013 ± 0.000 | 0.173 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.176 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.187 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.266 ± 0.002 | 0.073 ± 0.002 | 0.198 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.281 ± 0.003 | 0.055 ± 0.001 | 0.202 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.291 ± 0.010 | 0.062 ± 0.001 | 0.207 ± 0.003 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.113 ± 0.001 | 0.074 ± 0.001 | 0.193 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.122 ± 0.001 | 0.057 ± 0.000 | 0.200 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.129 ± 0.002 | 0.065 ± 0.001 | 0.205 ± 0.006 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.722 ± 0.014 | 0.076 ± 0.000 | 0.385 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.733 ± 0.005 | 0.049 ± 0.000 | 0.381 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.731 ± 0.003 | 0.049 ± 0.000 | 0.384 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.227 ± 0.003 | 0.072 ± 0.001 | 0.336 ± 0.008 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.231 ± 0.004 | 0.047 ± 0.001 | 0.333 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.235 ± 0.004 | 0.048 ± 0.000 | 0.335 ± 0.009 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.072 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.075 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.078 ± 0.004 | 0.017 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.988 ± 0.006 | 0.120 ± 0.001 | 0.289 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.994 ± 0.016 | 0.080 ± 0.001 | 0.285 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.994 ± 0.005 | 0.083 ± 0.001 | 0.290 ± 0.005 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.215 ± 0.001 | 0.123 ± 0.001 | 0.322 ± 0.057 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.219 ± 0.005 | 0.083 ± 0.000 | 0.315 ± 0.008 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.223 ± 0.002 | 0.087 ± 0.001 | 0.320 ± 0.007 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.079 ± 0.000 | 0.121 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.001 | 0.124 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.052 ± 0.001 | 0.124 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.199 ± 0.001 | 0.055 ± 0.000 | 0.405 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.201 ± 0.001 | 0.035 ± 0.000 | 0.404 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.204 ± 0.002 | 0.037 ± 0.000 | 0.413 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.109 ± 0.001 | 0.056 ± 0.001 | 0.244 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.109 ± 0.001 | 0.035 ± 0.000 | 0.245 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.112 ± 0.002 | 0.036 ± 0.000 | 0.249 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.108 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.001 | 0.116 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.000 | 0.127 ± 0.001 | 0.031 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.234 ± 0.030 | 0.049 ± 0.000 | 0.267 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.249 ± 0.001 | 0.040 ± 0.000 | 0.273 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.274 ± 0.002 | 0.054 ± 0.000 | 0.286 ± 0.004 | 0.026 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.103 ± 0.001 | 0.047 ± 0.000 | 0.214 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.111 ± 0.002 | 0.039 ± 0.000 | 0.221 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.124 ± 0.003 | 0.054 ± 0.000 | 0.233 ± 0.002 | 0.032 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.030 ± 0.000 | 0.032 ± 0.000 | 0.016 ± 7.136 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.050 ± 0.000 | 0.050 ± 0.000 | 0.033 ± 0.002 | 0.037 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 31.1x faster than the slowest successful cell (`jax-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 28.5x faster than the slowest successful cell (`jax-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 30.3x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.0x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-trace`, 303.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 303.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 169.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 35.3x faster than the slowest successful cell (`tenferro-eager`, 169.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 169.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 440.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 390.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 76.0x faster than the slowest successful cell (`tenferro-eager`, 290.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 98.2x faster than the slowest successful cell (`tenferro-eager`, 290.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 78.9x faster than the slowest successful cell (`tenferro-trace`, 290.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 60.9x faster than the slowest successful cell (`tenferro-trace`, 290.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 43.1x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 42.9x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 44.0x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.7x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 54.1x faster than the slowest successful cell (`tenferro-eager`, 0.722 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 49.1x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 37.2x faster than the slowest successful cell (`tenferro-eager`, 0.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.731 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.3x faster than the slowest successful cell (`pytorch-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.0x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.335 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 72.5x faster than the slowest successful cell (`tenferro-eager`, 0.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.3x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 45.9x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.320 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.405 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.7x faster than the slowest successful cell (`pytorch-cpu`, 0.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.4x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.267 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

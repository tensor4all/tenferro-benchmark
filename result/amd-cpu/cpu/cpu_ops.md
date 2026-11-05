# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `ce2efd37d9e132703dfd2ebefbf4353249cf53ea`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261105_180410/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261105_180410/run.yaml`
- Timestamp: `2026-11-05T18:04:10.102563+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261105_180410`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261105_180410/cpu_ops_t1_20261105_180410.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.047 ± 0.001 | 0.043 ± 0.000 | 0.033 ± 0.023 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.041 ± 0.000 | 0.041 ± 0.000 | 0.044 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.144 ± 0.001 | 0.144 ± 0.000 | 0.134 ± 0.002 | 0.146 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.176 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.220 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.150 ± 0.002 | 0.153 ± 0.001 | 0.180 ± 0.003 | 0.202 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.189 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.219 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.044 ± 0.000 | 0.041 ± 0.001 | 0.075 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.028 ± 0.000 | 0.049 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.001 | 0.028 ± 0.000 | 0.029 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.090 ± 0.000 | 0.080 ± 0.001 | 0.096 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.083 ± 0.000 | 0.082 ± 0.000 | 0.077 ± 0.001 | 0.076 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.307 ± 0.006 | 0.304 ± 0.002 | 0.267 ± 0.005 | 0.251 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.144 ± 0.003 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.149 ± 0.003 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.150 ± 0.002 | 0.015 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.177 ± 0.004 | 0.036 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.000 | 0.137 ± 0.004 | 0.026 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.000 | 0.153 ± 0.005 | 0.041 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.141 ± 0.003 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.001 | 0.159 ± 0.003 | 0.057 ± 0.003 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.307 ± 0.006 | 0.302 ± 0.001 | 0.325 ± 0.006 | 0.397 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 16.411 ± 0.157 | 8.655 ± 0.122 | 10.439 ± 0.140 | 9.053 ± 0.031 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 105.398 ± 0.718 | 55.419 ± 0.375 | 66.302 ± 0.327 | 57.530 ± 0.289 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.594 ± 0.098 | 8.706 ± 0.130 | 8.631 ± 0.112 | 9.079 ± 0.036 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 62.894 ± 0.503 | 55.295 ± 0.104 | 52.089 ± 0.536 | 57.539 ± 0.119 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 6.226 ± 0.049 | 5.113 ± 0.087 | 13.369 ± 0.460 | 4.469 ± 0.070 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 40.987 ± 0.454 | 35.795 ± 0.118 | 81.045 ± 1.368 | 32.020 ± 0.566 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 5.837 ± 0.110 | 5.324 ± 0.023 | 7.269 ± 0.137 | 4.387 ± 0.087 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 39.008 ± 0.327 | 36.163 ± 0.882 | 43.558 ± 0.460 | 33.520 ± 0.608 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.042 ± 0.001 | 0.026 ± 0.000 | 0.034 ± 0.001 | 0.036 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.001 | 0.159 ± 0.003 | 0.092 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 11.175 ± 0.109 | 7.324 ± 0.116 | 7.440 ± 0.101 | 7.213 ± 0.039 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 76.838 ± 0.216 | 53.930 ± 0.422 | 51.704 ± 0.322 | 46.622 ± 2.432 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 7.676 ± 0.107 | 7.051 ± 0.049 | 7.440 ± 0.102 | 7.327 ± 0.072 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 51.173 ± 0.185 | 50.042 ± 0.587 | 51.847 ± 0.575 | 53.523 ± 0.468 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.003 | 0.183 ± 0.006 | 0.059 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 2.107 ± 0.035 | 0.738 ± 0.001 | 1.247 ± 0.021 | 0.685 ± 0.015 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 9.539 ± 0.043 | 4.281 ± 0.053 | 5.790 ± 0.230 | 4.275 ± 0.066 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.081 ± 0.012 | 0.740 ± 0.003 | 2.236 ± 0.059 | 0.677 ± 0.008 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 5.174 ± 0.054 | 4.292 ± 0.018 | 9.664 ± 0.146 | 4.152 ± 0.032 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.655 ± 0.008 | 0.786 ± 0.021 | 0.658 ± 0.010 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 28.888 ± 0.338 | 14.993 ± 0.109 | 22.243 ± 0.480 | 15.726 ± 0.141 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 195.520 ± 0.502 | 105.091 ± 0.201 | 135.339 ± 0.603 | 110.161 ± 0.190 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 15.114 ± 0.055 | 14.977 ± 0.058 | 16.381 ± 0.042 | 15.802 ± 0.213 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 103.809 ± 0.661 | 106.126 ± 0.814 | 107.269 ± 0.441 | 117.176 ± 0.282 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.134 ± 0.004 | 0.130 ± 0.002 | 0.208 ± 0.001 | 0.201 ± 0.006 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.879 ± 0.016 | 0.866 ± 0.026 | 1.098 ± 0.353 | 1.497 ± 0.078 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.323 ± 0.040 | 3.396 ± 0.078 | 3.682 ± 0.117 | 4.745 ± 0.274 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.097 ± 0.001 | 0.095 ± 0.001 | 0.117 ± 0.001 | 0.112 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.034 ± 0.000 | 0.038 ± 0.000 | 0.053 ± 0.001 | 0.047 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.045 ± 0.002 | 0.050 ± 0.003 | 0.071 ± 0.002 | 0.082 ± 0.010 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.079 ± 0.003 | 0.086 ± 0.001 | 0.122 ± 0.006 | 0.161 ± 0.010 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.580 ± 0.003 | 0.582 ± 0.007 | 0.646 ± 0.009 | 0.594 ± 0.010 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.175 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.176 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.187 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.348 ± 0.003 | 0.035 ± 0.000 | 0.201 ± 0.012 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.361 ± 0.002 | 0.040 ± 0.001 | 0.203 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.370 ± 0.006 | 0.045 ± 0.000 | 0.209 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.205 ± 0.002 | 0.036 ± 0.001 | 0.202 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.214 ± 0.001 | 0.041 ± 0.001 | 0.203 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.220 ± 0.003 | 0.046 ± 0.000 | 0.211 ± 0.009 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.363 ± 0.003 | 0.032 ± 0.001 | 0.276 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.372 ± 0.066 | 0.032 ± 0.000 | 0.277 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.369 ± 0.004 | 0.034 ± 0.000 | 0.277 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.262 ± 0.002 | 0.031 ± 0.000 | 0.257 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.268 ± 0.001 | 0.032 ± 0.000 | 0.259 ± 0.005 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.270 ± 0.005 | 0.033 ± 0.000 | 0.263 ± 0.008 | 0.019 ± 0.005 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.072 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.074 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.074 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.710 ± 0.005 | 0.057 ± 0.000 | 0.220 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.706 ± 0.005 | 0.059 ± 0.001 | 0.218 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.714 ± 0.004 | 0.062 ± 0.001 | 0.222 ± 0.003 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.406 ± 0.008 | 0.062 ± 0.000 | 0.251 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.409 ± 0.006 | 0.062 ± 0.001 | 0.248 ± 0.004 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.415 ± 0.012 | 0.065 ± 0.000 | 0.251 ± 0.005 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.121 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.001 | 0.127 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.124 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.278 ± 0.002 | 0.021 ± 0.000 | 0.413 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.278 ± 0.001 | 0.021 ± 0.000 | 0.418 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.280 ± 0.002 | 0.021 ± 0.000 | 0.415 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.195 ± 0.002 | 0.021 ± 0.000 | 0.248 ± 0.007 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.196 ± 0.002 | 0.021 ± 0.000 | 0.247 ± 0.005 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.197 ± 0.003 | 0.022 ± 0.000 | 0.250 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.109 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.116 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.000 | 0.128 ± 0.003 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.312 ± 0.001 | 0.022 ± 0.000 | 0.266 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.332 ± 0.003 | 0.026 ± 0.000 | 0.273 ± 0.007 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.354 ± 0.012 | 0.038 ± 0.000 | 0.285 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.195 ± 0.004 | 0.021 ± 0.000 | 0.217 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.202 ± 0.006 | 0.027 ± 0.000 | 0.221 ± 0.005 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.215 ± 0.003 | 0.038 ± 0.000 | 0.236 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.023 ± 0.001 | 0.023 ± 0.000 | 0.043 ± 0.001 | 0.051 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.008 ± 0.002 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 32.8x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 27.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.9x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 50.9x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 48.2x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 50.7x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 31.2x faster than the slowest successful cell (`tenferro-eager`, 0.348 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 27.0x faster than the slowest successful cell (`tenferro-eager`, 0.361 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 28.0x faster than the slowest successful cell (`tenferro-eager`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 0.220 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 28.3x faster than the slowest successful cell (`tenferro-eager`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 28.7x faster than the slowest successful cell (`tenferro-eager`, 0.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 24.7x faster than the slowest successful cell (`tenferro-eager`, 0.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`tenferro-eager`, 0.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 18.1x faster than the slowest successful cell (`tenferro-eager`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.270 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 53.9x faster than the slowest successful cell (`tenferro-eager`, 0.710 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.710 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 51.8x faster than the slowest successful cell (`tenferro-eager`, 0.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 39.9x faster than the slowest successful cell (`tenferro-eager`, 0.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.714 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 32.4x faster than the slowest successful cell (`tenferro-eager`, 0.406 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 26.7x faster than the slowest successful cell (`tenferro-eager`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 28.8x faster than the slowest successful cell (`tenferro-eager`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 32.9x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 33.2x faster than the slowest successful cell (`pytorch-cpu`, 0.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.418 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 30.2x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.415 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.247 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 23.3x faster than the slowest successful cell (`tenferro-eager`, 0.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 0.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.221 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261105_184557/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261105_184557/run.yaml`
- Timestamp: `2026-11-05T18:45:57.792006+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261105_184557`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261105_184557/cpu_ops_t4_20261105_184557.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.047 ± 0.000 | 0.041 ± 0.000 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.043 ± 0.000 | 0.043 ± 0.000 | 0.040 ± 0.000 | 0.045 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.147 ± 0.001 | 0.146 ± 0.001 | 0.131 ± 0.001 | 0.139 ± 0.036 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.177 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.191 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.269 ± 0.000 | 0.269 ± 0.001 | 0.126 ± 0.002 | 0.241 ± 0.088 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.235 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.026 ± 0.001 | 0.193 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.024 ± 0.001 | 0.024 ± 0.000 | 0.032 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.045 ± 0.001 | 0.060 ± 0.001 | 0.098 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.032 ± 0.001 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.037 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.034 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.023 ± 0.000 | 0.040 ± 0.001 | 0.053 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.048 ± 0.000 | 0.047 ± 0.000 | 0.028 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.089 ± 0.000 | 0.078 ± 0.001 | 0.095 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.084 ± 0.000 | 0.083 ± 0.001 | 0.076 ± 0.000 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.305 ± 0.000 | 0.304 ± 0.004 | 0.263 ± 0.001 | 0.287 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.023 ± 0.000 | 0.143 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.001 | 0.147 ± 0.001 | 0.014 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.150 ± 0.003 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.045 ± 0.001 | 0.176 ± 0.007 | 0.061 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.049 ± 0.000 | 0.180 ± 0.002 | 0.026 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.001 | 0.190 ± 0.006 | 0.082 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.051 ± 0.000 | 0.182 ± 0.003 | 0.030 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.069 ± 0.000 | 0.202 ± 0.006 | 0.065 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.361 ± 0.026 | 0.346 ± 0.001 | 0.349 ± 0.012 | 0.518 ± 0.010 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 32.575 ± 10.339 | 22.737 ± 24.426 | 7.247 ± 0.178 | 10.300 ± 1.900 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 87.122 ± 1.652 | 81.998 ± 1.992 | 42.819 ± 1.237 | 55.886 ± 3.940 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 34.462 ± 14.446 | 18.993 ± 0.014 | 6.028 ± 0.149 | 10.535 ± 0.860 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 57.399 ± 0.580 | 48.036 ± 1.126 | 31.275 ± 1.516 | 55.778 ± 5.812 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 11.776 ± 0.046 | 19.509 ± 2.050 | 7.335 ± 0.064 | 5.439 ± 1.745 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 45.648 ± 9.732 | 51.225 ± 14.819 | 47.866 ± 0.399 | 28.486 ± 10.974 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 13.255 ± 5.775 | 17.998 ± 14.478 | 3.688 ± 0.019 | 5.727 ± 1.197 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 44.885 ± 4.425 | 49.542 ± 16.601 | 24.466 ± 0.168 | 33.735 ± 6.189 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.062 ± 0.001 | 0.035 ± 0.000 | 0.035 ± 0.001 | 0.038 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.095 ± 0.001 | 0.178 ± 0.001 | 0.082 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 67.395 ± 22.554 | 13.994 ± 18.664 | 5.074 ± 0.023 | 7.756 ± 0.804 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 153.446 ± 1.992 | 123.019 ± 2.005 | 34.129 ± 0.381 | 46.448 ± 12.801 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 33.835 ± 4.601 | 30.991 ± 33.999 | 5.398 ± 0.134 | 9.045 ± 0.881 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 125.589 ± 20.151 | 124.998 ± 3.998 | 32.662 ± 0.228 | 56.053 ± 6.888 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.122 ± 0.001 | 0.183 ± 0.002 | 0.099 ± 0.032 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 6.559 ± 0.123 | 2.999 ± 0.011 | 5.978 ± 17.865 | 0.873 ± 0.066 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 13.363 ± 4.464 | 4.926 ± 0.909 | 2.960 ± 0.087 | 3.909 ± 0.048 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 3.268 ± 0.058 | 2.689 ± 1.875 | 1.290 ± 0.039 | 0.977 ± 0.014 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 10.119 ± 2.332 | 4.787 ± 4.025 | 4.778 ± 0.045 | 3.840 ± 0.105 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.911 ± 0.004 | 1.001 ± 0.019 | 0.906 ± 0.048 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 42.844 ± 0.654 | 21.919 ± 0.254 | 19.354 ± 0.353 | 17.658 ± 2.213 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 170.486 ± 3.464 | 92.068 ± 1.463 | 94.819 ± 0.804 | 96.956 ± 2.930 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 54.012 ± 2.417 | 29.781 ± 16.285 | 15.129 ± 0.395 | 18.664 ± 0.586 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 118.522 ± 3.858 | 101.909 ± 4.265 | 83.619 ± 0.696 | 113.422 ± 7.342 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.108 ± 0.050 | 0.102 ± 0.001 | 0.162 ± 0.006 | 0.168 ± 0.025 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.498 ± 0.003 | 0.509 ± 0.012 | 0.711 ± 0.059 | 0.797 ± 0.140 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.880 ± 0.028 | 1.860 ± 0.028 | 2.669 ± 0.069 | 2.764 ± 0.149 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.181 ± 0.008 | 0.178 ± 0.001 | 0.184 ± 0.009 | 0.119 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.054 ± 0.000 | 0.063 ± 0.000 | 0.053 ± 0.000 | 0.060 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.072 ± 0.002 | 0.081 ± 0.002 | 0.097 ± 0.002 | 0.156 ± 0.038 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.092 ± 0.001 | 0.103 ± 0.002 | 0.136 ± 0.011 | 0.196 ± 0.026 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.819 ± 0.007 | 0.814 ± 0.021 | 0.846 ± 0.010 | 0.841 ± 0.014 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.176 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.180 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.231 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.366 ± 0.004 | 0.052 ± 0.000 | 0.195 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.379 ± 0.005 | 0.058 ± 0.001 | 0.204 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.388 ± 0.003 | 0.062 ± 0.001 | 0.207 ± 0.002 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.224 ± 0.001 | 0.053 ± 0.001 | 0.195 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.235 ± 0.002 | 0.059 ± 0.001 | 0.201 ± 0.004 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.242 ± 0.004 | 0.064 ± 0.002 | 0.211 ± 0.006 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.386 ± 0.006 | 0.052 ± 0.001 | 0.380 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.385 ± 0.007 | 0.053 ± 0.000 | 0.377 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.386 ± 0.005 | 0.055 ± 0.001 | 0.380 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.287 ± 0.004 | 0.050 ± 0.000 | 0.330 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.287 ± 0.004 | 0.051 ± 0.001 | 0.333 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.296 ± 0.003 | 0.052 ± 0.000 | 0.338 ± 0.009 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.072 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.074 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.075 ± 0.001 | 0.018 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.721 ± 0.003 | 0.087 ± 0.001 | 0.287 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.749 ± 0.004 | 0.089 ± 0.000 | 0.283 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.728 ± 0.006 | 0.091 ± 0.000 | 0.290 ± 0.003 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.432 ± 0.003 | 0.089 ± 0.001 | 0.316 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.434 ± 0.001 | 0.091 ± 0.002 | 0.318 ± 0.008 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.438 ± 0.004 | 0.095 ± 0.001 | 0.322 ± 0.012 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.119 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.123 ± 0.002 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.124 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.297 ± 0.004 | 0.033 ± 0.000 | 0.410 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.299 ± 0.003 | 0.034 ± 0.000 | 0.412 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.303 ± 0.003 | 0.034 ± 0.000 | 0.417 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.214 ± 0.001 | 0.034 ± 0.000 | 0.248 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.217 ± 0.002 | 0.034 ± 0.000 | 0.249 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.215 ± 0.000 | 0.034 ± 0.000 | 0.249 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.001 | 0.108 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.118 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.047 ± 0.000 | 0.127 ± 0.001 | 0.028 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.333 ± 0.002 | 0.031 ± 0.000 | 0.266 ± 0.005 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.345 ± 0.004 | 0.037 ± 0.001 | 0.269 ± 0.006 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.372 ± 0.002 | 0.047 ± 0.001 | 0.284 ± 0.005 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.208 ± 0.002 | 0.031 ± 0.000 | 0.216 ± 0.011 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.216 ± 0.003 | 0.036 ± 0.000 | 0.219 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.228 ± 0.003 | 0.046 ± 0.001 | 0.233 ± 0.004 | 0.033 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.029 ± 0.001 | 0.030 ± 0.000 | 0.035 ± 0.002 | 0.034 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.010 ± 0.003 | 0.013 ± 0.005 | 0.011 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.023 ± 0.000 | 0.023 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 32.9x faster than the slowest successful cell (`jax-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.5x faster than the slowest successful cell (`jax-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 67.395 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 37.5x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 47.4x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 31.4x faster than the slowest successful cell (`tenferro-eager`, 0.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 24.9x faster than the slowest successful cell (`tenferro-eager`, 0.379 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.388 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.224 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 29.6x faster than the slowest successful cell (`tenferro-eager`, 0.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 26.6x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-eager`, 0.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.338 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 55.4x faster than the slowest successful cell (`tenferro-eager`, 0.721 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 47.6x faster than the slowest successful cell (`tenferro-eager`, 0.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 34.0x faster than the slowest successful cell (`tenferro-eager`, 0.728 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 34.1x faster than the slowest successful cell (`tenferro-eager`, 0.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 27.7x faster than the slowest successful cell (`tenferro-eager`, 0.434 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 22.5x faster than the slowest successful cell (`tenferro-eager`, 0.438 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 31.1x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.9x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.417 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.7x faster than the slowest successful cell (`tenferro-eager`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.345 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Benchmark commit: `8eb4f087ca7cfae8e8f724cdb0b344ad5123e372`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261106_180658/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_180658/run.yaml`
- Timestamp: `2026-11-06T18:06:58.553919+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261106_180658`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261106_180658/cpu_ops_t1_20261106_180658.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.001 | 0.039 ± 0.000 | 0.037 ± 0.001 | 0.039 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.038 ± 0.001 | 0.038 ± 0.000 | 0.036 ± 0.001 | 0.038 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.135 ± 0.002 | 0.137 ± 0.002 | 0.119 ± 0.002 | 0.125 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.127 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.147 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.120 ± 0.002 | 0.121 ± 0.002 | 0.151 ± 0.004 | 0.153 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.140 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.138 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.001 | 0.017 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.037 ± 0.001 | 0.036 ± 0.000 | 0.048 ± 0.001 | 0.054 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.035 ± 0.007 | 0.024 ± 0.001 | 0.025 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.076 ± 0.003 | 0.076 ± 0.001 | 0.075 ± 0.001 | 0.082 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.077 ± 0.001 | 0.075 ± 0.001 | 0.068 ± 0.001 | 0.068 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.278 ± 0.009 | 0.275 ± 0.000 | 0.252 ± 0.007 | 0.231 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.087 ± 0.001 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.090 ± 0.002 | 0.026 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.091 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.001 | 0.111 ± 0.002 | 0.034 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.064 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.072 ± 0.001 | 0.043 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.066 ± 0.001 | 0.034 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.001 | 0.081 ± 0.000 | 0.057 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.225 ± 0.003 | 0.226 ± 0.004 | 0.221 ± 0.006 | 0.309 ± 0.013 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 9.902 ± 0.171 | 5.028 ± 0.061 | 5.780 ± 0.084 | 7.255 ± 0.125 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 54.751 ± 1.173 | 28.770 ± 0.342 | 33.418 ± 0.994 | 43.982 ± 0.529 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.624 ± 0.067 | 4.991 ± 0.229 | 4.817 ± 0.132 | 7.268 ± 0.253 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 33.117 ± 0.823 | 28.982 ± 0.542 | 26.740 ± 0.643 | 43.969 ± 0.261 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.369 ± 0.352 | 2.239 ± 0.054 | 4.852 ± 0.148 | 5.124 ± 0.055 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 19.949 ± 0.684 | 16.007 ± 0.187 | 39.689 ± 0.512 | 33.342 ± 0.464 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.929 ± 0.042 | 2.391 ± 0.012 | 2.970 ± 0.074 | 5.114 ± 0.122 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 18.267 ± 0.149 | 16.699 ± 0.143 | 21.832 ± 0.420 | 34.148 ± 0.395 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.023 ± 0.001 | 0.012 ± 0.001 | 0.019 ± 0.001 | 0.048 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.036 ± 0.001 | 0.078 ± 0.003 | 0.084 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 6.167 ± 0.076 | 3.495 ± 0.103 | 4.046 ± 0.121 | 6.642 ± 0.045 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 35.816 ± 0.828 | 24.312 ± 0.214 | 26.334 ± 0.325 | 39.192 ± 0.633 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 4.437 ± 0.036 | 3.600 ± 0.079 | 3.933 ± 0.074 | 6.614 ± 0.140 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 26.363 ± 0.440 | 25.086 ± 0.593 | 25.060 ± 0.208 | 45.017 ± 1.112 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.049 ± 0.001 | 0.095 ± 0.002 | 0.057 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.427 ± 0.021 | 0.387 ± 0.000 | 0.863 ± 0.026 | 0.593 ± 0.012 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 5.410 ± 0.146 | 2.193 ± 0.020 | 3.170 ± 0.043 | 3.584 ± 0.022 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.823 ± 0.018 | 0.397 ± 0.003 | 1.249 ± 0.079 | 0.736 ± 0.046 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.494 ± 0.189 | 2.301 ± 0.038 | 5.096 ± 0.165 | 3.898 ± 0.176 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.454 ± 0.005 | 0.549 ± 0.022 | 0.533 ± 0.014 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 17.420 ± 0.320 | 8.417 ± 0.278 | 10.482 ± 0.308 | 12.827 ± 1.341 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 107.637 ± 1.788 | 55.459 ± 0.383 | 70.184 ± 1.517 | 76.563 ± 1.691 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 9.167 ± 0.085 | 8.627 ± 0.271 | 8.860 ± 0.363 | 12.279 ± 0.132 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 55.843 ± 0.448 | 56.122 ± 0.494 | 49.281 ± 0.680 | 84.567 ± 0.613 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.057 ± 0.004 | 0.054 ± 0.001 | 0.108 ± 0.003 | 0.194 ± 0.005 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.416 ± 0.008 | 0.406 ± 0.007 | 0.604 ± 0.009 | 1.304 ± 0.031 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 1.726 ± 0.012 | 1.714 ± 0.012 | 1.903 ± 0.135 | 4.765 ± 0.190 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.053 ± 0.003 | 0.050 ± 0.003 | 0.065 ± 0.002 | 0.086 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.028 ± 0.001 | 0.029 ± 0.000 | 0.039 ± 0.001 | 0.048 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.031 ± 0.002 | 0.032 ± 0.001 | 0.049 ± 0.003 | 0.057 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.039 ± 0.000 | 0.043 ± 0.000 | 0.082 ± 0.023 | 0.134 ± 0.006 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.437 ± 0.006 | 0.446 ± 0.011 | 0.474 ± 0.012 | 0.491 ± 0.022 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.123 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.134 ± 0.007 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.136 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.274 ± 0.007 | 0.019 ± 0.000 | 0.111 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.443 ± 0.066 | 0.021 ± 0.000 | 0.109 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.303 ± 0.005 | 0.026 ± 0.001 | 0.117 ± 0.006 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.122 ± 0.002 | 0.019 ± 0.000 | 0.100 ± 0.004 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.129 ± 0.003 | 0.021 ± 0.000 | 0.103 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.136 ± 0.003 | 0.025 ± 0.001 | 0.108 ± 0.002 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.295 ± 0.003 | 0.023 ± 0.001 | 0.169 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.303 ± 0.006 | 0.023 ± 0.001 | 0.163 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.310 ± 0.007 | 0.024 ± 0.001 | 0.167 ± 0.003 | 0.026 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.189 ± 0.006 | 0.022 ± 0.000 | 0.147 ± 0.008 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.203 ± 0.040 | 0.023 ± 0.001 | 0.143 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.196 ± 0.002 | 0.023 ± 0.000 | 0.144 ± 0.001 | 0.025 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.035 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.037 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.036 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.679 ± 0.007 | 0.034 ± 0.001 | 0.125 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.689 ± 0.008 | 0.034 ± 0.000 | 0.126 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.701 ± 0.021 | 0.035 ± 0.001 | 0.125 ± 0.002 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.347 ± 0.009 | 0.032 ± 0.002 | 0.133 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.358 ± 0.004 | 0.033 ± 0.001 | 0.134 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.369 ± 0.007 | 0.034 ± 0.001 | 0.135 ± 0.003 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.054 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.057 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.057 ± 0.002 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.205 ± 0.006 | 0.015 ± 0.000 | 0.275 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.212 ± 0.084 | 0.015 ± 0.000 | 0.277 ± 0.011 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.212 ± 0.005 | 0.016 ± 0.000 | 0.275 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.113 ± 0.003 | 0.015 ± 0.000 | 0.128 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.115 ± 0.002 | 0.015 ± 0.000 | 0.128 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.117 ± 0.003 | 0.015 ± 0.001 | 0.130 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.043 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.049 ± 0.002 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.058 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.241 ± 0.005 | 0.015 ± 0.000 | 0.155 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.260 ± 0.003 | 0.018 ± 0.000 | 0.160 ± 0.007 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.285 ± 0.010 | 0.027 ± 0.001 | 0.171 ± 0.004 | 0.027 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.114 ± 0.004 | 0.014 ± 0.001 | 0.112 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.124 ± 0.002 | 0.017 ± 0.001 | 0.115 ± 0.003 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.134 ± 0.000 | 0.027 ± 0.000 | 0.121 ± 0.001 | 0.028 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.004 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.024 ± 0.000 | 0.040 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.018 ± 0.001 | 0.022 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 30.2x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 28.1x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 42.4x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 44.6x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 44.0x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 28.4x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 39.4x faster than the slowest successful cell (`tenferro-eager`, 0.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`tenferro-eager`, 0.443 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`tenferro-eager`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 0.295 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.295 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 26.2x faster than the slowest successful cell (`tenferro-eager`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-eager`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 63.0x faster than the slowest successful cell (`tenferro-eager`, 0.679 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.679 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 59.5x faster than the slowest successful cell (`tenferro-eager`, 0.689 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`tenferro-eager`, 0.689 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 30.4x faster than the slowest successful cell (`tenferro-eager`, 0.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 33.4x faster than the slowest successful cell (`tenferro-eager`, 0.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`tenferro-eager`, 0.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 29.9x faster than the slowest successful cell (`tenferro-eager`, 0.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.358 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.9x faster than the slowest successful cell (`tenferro-eager`, 0.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 0.369 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 25.6x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 24.8x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.3x faster than the slowest successful cell (`tenferro-eager`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`tenferro-eager`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261106_184117/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261106_184117/run.yaml`
- Timestamp: `2026-11-06T18:41:17.633321+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261106_184117`.

- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

### CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261106_184117/cpu_ops_t4_20261106_184117.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.001 | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.039 ± 0.000 | 0.039 ± 0.000 | 0.037 ± 0.001 | 0.044 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.039 ± 0.001 | 0.039 ± 0.000 | 0.036 ± 0.001 | 0.039 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.135 ± 0.003 | 0.136 ± 0.002 | 0.118 ± 0.003 | 0.155 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.126 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.158 ± 0.030 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.233 ± 0.005 | 0.231 ± 0.010 | 0.111 ± 0.002 | 0.295 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.176 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.000 | 0.180 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.027 ± 0.001 | 0.027 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.023 ± 0.001 | 0.022 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.039 ± 0.001 | 0.038 ± 0.000 | 0.064 ± 0.001 | 0.066 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.024 ± 0.001 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.029 ± 0.000 | 0.030 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.026 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.033 ± 0.001 | 0.049 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.025 ± 0.014 | 0.032 ± 0.015 | 0.024 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.076 ± 0.001 | 0.074 ± 0.001 | 0.072 ± 0.002 | 0.083 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.076 ± 0.004 | 0.075 ± 0.001 | 0.068 ± 0.001 | 0.067 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.280 ± 0.004 | 0.274 ± 0.001 | 0.246 ± 0.006 | 0.237 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.001 | 0.087 ± 0.002 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.091 ± 0.002 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.093 ± 0.002 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.003 | 0.113 ± 0.001 | 0.058 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.032 ± 0.001 | 0.090 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.042 ± 0.000 | 0.101 ± 0.001 | 0.070 ± 0.014 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.095 ± 0.002 | 0.031 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.001 | 0.109 ± 0.002 | 0.066 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.255 ± 0.001 | 0.241 ± 0.009 | 0.225 ± 0.002 | 0.346 ± 0.016 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 25.956 ± 0.169 | 13.998 ± 0.035 | 4.778 ± 7.769 | 8.886 ± 0.844 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 53.513 ± 0.947 | 60.583 ± 11.208 | 23.314 ± 0.736 | 46.113 ± 1.272 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 17.199 ± 0.082 | 14.007 ± 1.008 | 4.069 ± 0.187 | 9.984 ± 2.054 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 65.749 ± 4.662 | 30.400 ± 2.304 | 17.432 ± 0.617 | 41.417 ± 9.239 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 23.094 ± 40.789 | 18.862 ± 11.204 | 3.568 ± 0.105 | 4.956 ± 1.502 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 60.824 ± 16.808 | 36.840 ± 18.204 | 23.519 ± 0.219 | 34.058 ± 8.580 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 16.837 ± 5.486 | 17.987 ± 19.697 | 2.577 ± 0.051 | 5.988 ± 2.767 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 27.493 ± 4.649 | 30.414 ± 50.109 | 12.821 ± 0.348 | 35.067 ± 5.841 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.101 ± 0.008 | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.043 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.054 ± 0.001 | 0.093 ± 0.006 | 0.105 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 69.003 ± 54.683 | 20.017 ± 0.049 | 3.127 ± 0.050 | 8.216 ± 1.951 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 46.957 ± 6.944 | 30.103 ± 41.253 | 15.124 ± 0.066 | 41.706 ± 3.227 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 20.816 ± 0.057 | 20.013 ± 0.998 | 3.043 ± 0.070 | 8.815 ± 3.252 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 30.396 ± 2.745 | 31.020 ± 25.827 | 14.573 ± 0.225 | 51.163 ± 9.736 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.073 ± 0.002 | 0.099 ± 0.002 | 0.064 ± 0.013 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 8.628 ± 1.998 | 14.016 ± 4.940 | 0.878 ± 0.053 | 1.137 ± 0.436 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 18.192 ± 2.723 | 10.011 ± 9.003 | 2.118 ± 0.046 | 3.636 ± 0.536 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 6.156 ± 8.260 | 15.463 ± 2.962 | 1.008 ± 0.032 | 0.905 ± 0.058 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 16.650 ± 2.451 | 10.016 ± 5.993 | 3.151 ± 0.150 | 3.407 ± 0.157 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.634 ± 0.005 | 0.641 ± 0.020 | 0.872 ± 0.073 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 46.799 ± 7.675 | 15.315 ± 0.248 | 9.903 ± 0.175 | 14.831 ± 1.213 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 102.443 ± 6.127 | 50.632 ± 6.455 | 51.274 ± 0.781 | 84.958 ± 5.077 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 20.854 ± 0.021 | 22.422 ± 3.049 | 8.608 ± 0.217 | 20.549 ± 3.851 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 61.993 ± 18.658 | 60.449 ± 4.065 | 38.924 ± 0.733 | 82.384 ± 13.050 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.042 ± 0.001 | 0.038 ± 0.001 | 0.072 ± 0.004 | 0.171 ± 0.015 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.226 ± 0.002 | 0.218 ± 0.001 | 0.325 ± 0.014 | 0.730 ± 0.103 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.901 ± 0.032 | 0.905 ± 0.033 | 1.136 ± 0.039 | 2.587 ± 0.214 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.090 ± 0.001 | 0.087 ± 0.003 | 0.092 ± 0.003 | 0.106 ± 0.025 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.037 ± 0.002 | 0.039 ± 0.001 | 0.040 ± 0.003 | 0.071 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.047 ± 0.001 | 0.050 ± 0.002 | 0.070 ± 0.004 | 0.141 ± 0.007 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.061 ± 0.001 | 0.067 ± 0.005 | 0.092 ± 0.005 | 0.222 ± 0.015 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.580 ± 0.004 | 0.562 ± 0.007 | 0.523 ± 0.021 | 0.853 ± 0.046 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.125 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.128 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.175 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.292 ± 0.008 | 0.026 ± 0.000 | 0.108 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.309 ± 0.011 | 0.028 ± 0.001 | 0.112 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.321 ± 0.007 | 0.033 ± 0.001 | 0.118 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.136 ± 0.010 | 0.025 ± 0.001 | 0.101 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.139 ± 0.002 | 0.027 ± 0.001 | 0.102 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.148 ± 0.011 | 0.031 ± 0.001 | 0.108 ± 0.002 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.316 ± 0.011 | 0.032 ± 0.001 | 0.235 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.319 ± 0.009 | 0.032 ± 0.001 | 0.233 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.327 ± 0.020 | 0.032 ± 0.001 | 0.234 ± 0.004 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.208 ± 0.006 | 0.029 ± 0.001 | 0.182 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.207 ± 0.003 | 0.030 ± 0.001 | 0.182 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.212 ± 0.005 | 0.030 ± 0.000 | 0.183 ± 0.006 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.035 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.037 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.037 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.716 ± 0.005 | 0.045 ± 0.001 | 0.168 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.716 ± 0.002 | 0.045 ± 0.001 | 0.167 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.726 ± 0.014 | 0.047 ± 0.002 | 0.169 ± 0.006 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.374 ± 0.009 | 0.044 ± 0.001 | 0.171 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.383 ± 0.011 | 0.044 ± 0.001 | 0.169 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.385 ± 0.014 | 0.046 ± 0.001 | 0.173 ± 0.004 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.001 | 0.055 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.001 | 0.057 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.058 ± 0.002 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.219 ± 0.004 | 0.021 ± 0.001 | 0.271 ± 0.009 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.217 ± 0.005 | 0.021 ± 0.001 | 0.278 ± 0.017 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.227 ± 0.008 | 0.021 ± 0.001 | 0.272 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.127 ± 0.002 | 0.020 ± 0.000 | 0.129 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.129 ± 0.005 | 0.020 ± 0.000 | 0.126 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.129 ± 0.001 | 0.021 ± 0.001 | 0.130 ± 0.004 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.043 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.049 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.057 ± 0.002 | 0.026 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.252 ± 0.002 | 0.018 ± 0.000 | 0.159 ± 0.006 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.274 ± 0.014 | 0.021 ± 0.001 | 0.161 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.292 ± 0.005 | 0.031 ± 0.001 | 0.173 ± 0.005 | 0.025 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.121 ± 0.005 | 0.016 ± 0.000 | 0.111 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.130 ± 0.006 | 0.020 ± 0.000 | 0.112 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.143 ± 0.002 | 0.030 ± 0.001 | 0.122 ± 0.002 | 0.025 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.027 ± 0.001 | 0.039 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.008 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.020 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.022 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 69.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-trace`, 14.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 14.016 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-trace`, 15.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 15.3x faster than the slowest successful cell (`tenferro-trace`, 15.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.4x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 33.1x faster than the slowest successful cell (`jax-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 45.2x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 26.1x faster than the slowest successful cell (`tenferro-eager`, 0.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`tenferro-eager`, 0.309 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.2x faster than the slowest successful cell (`tenferro-eager`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 28.7x faster than the slowest successful cell (`tenferro-eager`, 0.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`tenferro-eager`, 0.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 26.3x faster than the slowest successful cell (`tenferro-eager`, 0.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.0x faster than the slowest successful cell (`tenferro-eager`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 65.6x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 57.6x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 33.6x faster than the slowest successful cell (`tenferro-eager`, 0.726 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.726 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 36.4x faster than the slowest successful cell (`tenferro-eager`, 0.374 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 31.9x faster than the slowest successful cell (`tenferro-eager`, 0.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.130 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.9x faster than the slowest successful cell (`tenferro-eager`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.252 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

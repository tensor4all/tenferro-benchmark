# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Benchmark commit: `9b03dd718025b3350d93eaab13a8b93dcc66cab2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261126_175240/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_175240/run.yaml`
- Timestamp: `2026-11-26T17:52:40.572826+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261126_175240`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261126_175240/cpu_ops_t1_20261126_175240.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.041 ± 0.001 | 0.032 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.040 ± 0.000 | 0.043 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.134 ± 0.000 | 0.133 ± 0.000 | 0.134 ± 0.001 | 0.128 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | 0.176 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.000 | 0.186 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.229 ± 0.002 | 0.229 ± 0.002 | 0.175 ± 0.002 | 0.199 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.186 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.001 | 0.188 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.020 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.046 ± 0.000 | 0.040 ± 0.001 | 0.060 ± 0.020 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.019 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.000 | 0.023 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.027 ± 0.001 | 0.029 ± 0.012 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.029 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.068 ± 0.000 | 0.068 ± 0.000 | 0.081 ± 0.000 | 0.095 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.056 ± 0.000 | 0.055 ± 0.000 | 0.076 ± 0.002 | 0.076 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.212 ± 0.001 | 0.210 ± 0.001 | 0.263 ± 0.001 | 0.248 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.086 ± 0.000 | 0.023 ± 0.000 | 0.144 ± 0.004 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.120 ± 0.001 | 0.055 ± 0.000 | 0.151 ± 0.005 | 0.016 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.088 ± 0.002 | 0.023 ± 0.000 | 0.149 ± 0.001 | 0.020 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.126 ± 0.009 | 0.056 ± 0.000 | 0.175 ± 0.008 | 0.034 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.130 ± 0.006 | 0.055 ± 0.000 | 0.135 ± 0.006 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.154 ± 0.000 | 0.094 ± 0.001 | 0.142 ± 0.007 | 0.039 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.136 ± 0.002 | 0.061 ± 0.000 | 0.137 ± 0.004 | 0.024 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.163 ± 0.007 | 0.118 ± 0.001 | 0.153 ± 0.002 | 0.054 ± 0.002 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.358 ± 0.002 | 0.356 ± 0.001 | 0.321 ± 0.007 | 0.385 ± 0.011 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 16.280 ± 0.227 | 12.031 ± 0.128 | 10.255 ± 0.162 | 8.995 ± 0.038 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 88.872 ± 0.393 | 65.132 ± 0.406 | 65.601 ± 0.626 | 57.472 ± 0.302 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.056 ± 0.023 | 12.299 ± 0.034 | 8.446 ± 0.148 | 8.901 ± 0.071 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 54.203 ± 0.458 | 65.346 ± 0.362 | 51.701 ± 0.462 | 56.722 ± 0.237 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 9.618 ± 0.026 | 4.814 ± 0.018 | 12.255 ± 0.045 | 4.347 ± 0.076 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 70.834 ± 0.262 | 33.747 ± 0.074 | 79.702 ± 1.759 | 31.273 ± 0.738 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 8.812 ± 0.133 | 5.095 ± 0.014 | 5.384 ± 0.180 | 4.380 ± 0.038 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 69.137 ± 0.248 | 38.902 ± 0.643 | 42.388 ± 0.459 | 32.779 ± 0.853 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.029 ± 0.001 | 0.019 ± 0.000 | 0.033 ± 0.001 | 0.041 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.137 ± 0.006 | 0.055 ± 0.000 | 0.151 ± 0.007 | 0.081 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 14.080 ± 0.198 | 8.211 ± 0.045 | 7.376 ± 0.274 | 7.075 ± 0.068 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 109.374 ± 0.833 | 57.585 ± 0.286 | 51.516 ± 0.767 | 45.341 ± 2.433 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 11.225 ± 0.047 | 8.210 ± 0.063 | 7.263 ± 0.096 | 7.208 ± 0.069 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 91.691 ± 2.703 | 57.603 ± 0.280 | 50.472 ± 0.578 | 52.675 ± 3.036 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.169 ± 0.009 | 0.130 ± 0.005 | 0.171 ± 0.003 | 0.055 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.310 ± 0.007 | 1.080 ± 0.008 | 1.237 ± 0.037 | 0.668 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.660 ± 0.031 | 6.356 ± 0.159 | 5.441 ± 0.009 | 4.135 ± 0.060 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.712 ± 0.009 | 1.088 ± 0.003 | 1.730 ± 0.007 | 0.664 ± 0.006 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.510 ± 0.108 | 6.282 ± 0.040 | 9.426 ± 0.090 | 4.069 ± 0.077 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.725 ± 0.012 | 0.858 ± 0.009 | 0.786 ± 0.018 | 0.653 ± 0.007 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 27.086 ± 0.162 | 20.519 ± 0.282 | 20.814 ± 0.100 | 15.558 ± 0.109 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 177.903 ± 0.726 | 130.345 ± 0.488 | 134.894 ± 1.347 | 106.551 ± 0.542 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.267 ± 0.076 | 20.165 ± 0.103 | 15.849 ± 0.126 | 15.484 ± 0.075 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 94.215 ± 0.219 | 128.410 ± 0.805 | 107.340 ± 1.243 | 113.245 ± 0.927 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.096 ± 0.001 | 0.097 ± 0.001 | 0.195 ± 0.005 | 0.184 ± 0.003 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.717 ± 0.005 | 0.718 ± 0.002 | 1.068 ± 0.027 | 1.252 ± 0.048 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.027 ± 0.062 | 3.033 ± 0.017 | 3.536 ± 0.073 | 4.637 ± 0.153 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.112 ± 0.001 | 0.110 ± 0.000 | 0.113 ± 0.002 | 0.104 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.034 ± 0.000 | 0.052 ± 0.001 | 0.042 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.037 ± 0.002 | 0.040 ± 0.000 | 0.067 ± 0.002 | 0.066 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.070 ± 0.000 | 0.053 ± 0.000 | 0.124 ± 0.001 | 0.158 ± 0.034 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.564 ± 0.034 | 0.562 ± 0.007 | 0.636 ± 0.003 | 0.581 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.172 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.175 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.191 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.256 ± 0.004 | 0.035 ± 0.000 | 0.194 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.265 ± 0.002 | 0.041 ± 0.001 | 0.199 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.277 ± 0.003 | 0.050 ± 0.001 | 0.204 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.098 ± 0.002 | 0.036 ± 0.000 | 0.194 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.101 ± 0.001 | 0.041 ± 0.000 | 0.199 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.107 ± 0.001 | 0.051 ± 0.001 | 0.202 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.696 ± 0.005 | 0.034 ± 0.000 | 0.277 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.694 ± 0.006 | 0.034 ± 0.000 | 0.275 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.723 ± 0.003 | 0.037 ± 0.000 | 0.275 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.179 ± 0.003 | 0.033 ± 0.002 | 0.259 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.178 ± 0.002 | 0.033 ± 0.001 | 0.255 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.181 ± 0.004 | 0.036 ± 0.000 | 0.259 ± 0.007 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.074 ± 0.001 | 0.012 ± 0.000 | 0.071 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.089 ± 0.016 | 0.012 ± 0.000 | 0.073 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.075 ± 0.001 | 0.012 ± 0.000 | 0.074 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.816 ± 0.007 | 0.054 ± 0.001 | 0.220 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.833 ± 0.004 | 0.056 ± 0.001 | 0.218 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.841 ± 0.008 | 0.060 ± 0.001 | 0.221 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.177 ± 0.005 | 0.058 ± 0.000 | 0.245 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.174 ± 0.001 | 0.060 ± 0.000 | 0.245 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.179 ± 0.001 | 0.065 ± 0.007 | 0.249 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.120 ± 0.011 | 0.040 ± 0.000 | 0.119 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.120 ± 0.001 | 0.041 ± 0.000 | 0.122 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.123 ± 0.005 | 0.043 ± 0.001 | 0.123 ± 0.002 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.191 ± 0.002 | 0.028 ± 0.000 | 0.409 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.192 ± 0.004 | 0.029 ± 0.000 | 0.410 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.196 ± 0.004 | 0.031 ± 0.000 | 0.410 ± 0.005 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.101 ± 0.002 | 0.029 ± 0.000 | 0.245 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.101 ± 0.002 | 0.029 ± 0.001 | 0.244 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.104 ± 0.001 | 0.031 ± 0.000 | 0.243 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.101 ± 0.001 | 0.022 ± 0.000 | 0.106 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.106 ± 0.004 | 0.027 ± 0.000 | 0.114 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.114 ± 0.006 | 0.038 ± 0.000 | 0.125 ± 0.003 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.224 ± 0.002 | 0.022 ± 0.000 | 0.263 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.229 ± 0.002 | 0.027 ± 0.000 | 0.268 ± 0.039 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.248 ± 0.004 | 0.038 ± 0.000 | 0.285 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.087 ± 0.001 | 0.022 ± 0.000 | 0.213 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.093 ± 0.002 | 0.027 ± 0.000 | 0.216 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.099 ± 0.000 | 0.038 ± 0.000 | 0.229 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.040 ± 0.001 | 0.038 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 25.5x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 48.6x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 49.1x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 52.4x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 22.0x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 52.6x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 50.9x faster than the slowest successful cell (`tenferro-eager`, 0.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 48.8x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 62.1x faster than the slowest successful cell (`tenferro-eager`, 0.816 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.816 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 59.7x faster than the slowest successful cell (`tenferro-eager`, 0.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.833 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 47.4x faster than the slowest successful cell (`tenferro-eager`, 0.841 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.841 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 31.7x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.409 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 30.5x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 28.3x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.285 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261126_183225/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261126_183225/run.yaml`
- Timestamp: `2026-11-26T18:32:25.628209+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261126_183225`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/amd-cpu/cpu/einsum/20261126_183225/cpu_ops_t4_20261126_183225.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.035 ± 0.001 | 0.034 ± 0.000 | 0.041 ± 0.001 | 0.047 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.036 ± 0.000 | 0.035 ± 0.001 | 0.040 ± 0.001 | 0.044 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.131 ± 0.001 | 0.130 ± 0.001 | 0.131 ± 0.001 | 0.133 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.179 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.227 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.235 ± 0.001 | 0.227 ± 0.000 | 0.124 ± 0.001 | 0.339 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.193 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.025 ± 0.000 | 0.192 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.032 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.047 ± 0.001 | 0.046 ± 0.000 | 0.060 ± 0.001 | 0.092 ± 0.027 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.031 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.037 ± 0.000 | 0.032 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.033 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.041 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.028 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.068 ± 0.000 | 0.068 ± 0.000 | 0.077 ± 0.002 | 0.095 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.057 ± 0.000 | 0.056 ± 0.000 | 0.075 ± 0.001 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.211 ± 0.000 | 0.209 ± 0.000 | 0.260 ± 0.001 | 0.258 ± 0.023 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.093 ± 0.001 | 0.025 ± 0.000 | 0.143 ± 0.003 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.127 ± 0.010 | 0.058 ± 0.000 | 0.146 ± 0.003 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.101 ± 0.009 | 0.025 ± 0.000 | 0.149 ± 0.003 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.132 ± 0.011 | 0.058 ± 0.000 | 0.171 ± 0.002 | 0.078 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.140 ± 0.003 | 0.067 ± 0.000 | 0.176 ± 0.001 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.159 ± 0.001 | 0.105 ± 0.000 | 0.187 ± 0.005 | 0.076 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.158 ± 0.031 | 0.072 ± 0.001 | 0.182 ± 0.002 | 0.031 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.169 ± 0.020 | 0.126 ± 0.001 | 0.198 ± 0.002 | 0.065 ± 0.004 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.357 ± 0.002 | 0.352 ± 0.003 | 0.333 ± 0.006 | 0.521 ± 0.005 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 18.236 ± 0.088 | 14.268 ± 0.249 | 7.161 ± 0.104 | 10.333 ± 1.617 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 88.518 ± 0.395 | 64.315 ± 0.475 | 41.759 ± 2.749 | 55.920 ± 7.338 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 11.037 ± 0.096 | 14.251 ± 0.027 | 5.929 ± 0.019 | 10.957 ± 0.864 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 49.667 ± 1.043 | 64.426 ± 0.189 | 29.446 ± 1.808 | 56.880 ± 2.192 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 7.455 ± 0.049 | 3.869 ± 0.045 | 7.179 ± 0.152 | 4.445 ± 3.566 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 39.715 ± 0.176 | 21.020 ± 0.052 | 46.903 ± 0.359 | 32.890 ± 4.151 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 6.139 ± 0.072 | 3.922 ± 0.012 | 3.646 ± 0.075 | 4.006 ± 0.850 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 36.735 ± 1.082 | 23.591 ± 0.215 | 24.510 ± 0.258 | 31.925 ± 4.816 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.051 ± 0.001 | 0.038 ± 0.001 | 0.037 ± 0.001 | 0.038 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.220 ± 0.019 | 0.104 ± 0.001 | 0.176 ± 0.008 | 0.096 ± 0.009 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 13.528 ± 0.293 | 9.111 ± 0.044 | 4.897 ± 0.244 | 8.930 ± 4.454 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 57.640 ± 3.613 | 33.652 ± 0.582 | 33.495 ± 0.836 | 54.959 ± 13.108 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 9.116 ± 0.075 | 9.181 ± 0.095 | 5.070 ± 0.105 | 9.671 ± 0.613 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 47.815 ± 0.758 | 33.759 ± 0.199 | 32.563 ± 0.796 | 52.962 ± 6.966 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.217 ± 0.018 | 0.137 ± 0.005 | 0.171 ± 0.004 | 0.090 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.472 ± 0.008 | 1.232 ± 0.023 | 5.020 ± 5.572 | 0.905 ± 0.034 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.260 ± 0.082 | 4.988 ± 0.119 | 2.993 ± 0.106 | 3.772 ± 0.092 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.848 ± 0.003 | 1.236 ± 0.015 | 1.640 ± 3.980 | 0.955 ± 0.135 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.878 ± 0.104 | 5.019 ± 0.059 | 4.736 ± 0.091 | 10.053 ± 4.690 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.017 ± 0.014 | 1.205 ± 0.005 | 0.993 ± 0.021 | 0.895 ± 0.017 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 29.055 ± 0.059 | 22.062 ± 0.065 | 16.551 ± 0.368 | 17.988 ± 1.009 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 174.194 ± 1.323 | 124.191 ± 1.217 | 93.037 ± 3.291 | 102.269 ± 3.077 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 16.449 ± 0.136 | 21.993 ± 0.113 | 14.648 ± 0.389 | 17.988 ± 1.755 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 88.247 ± 0.918 | 123.468 ± 0.308 | 81.867 ± 2.821 | 106.948 ± 5.227 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.073 ± 0.000 | 0.073 ± 0.000 | 0.146 ± 0.003 | 0.173 ± 0.020 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.395 ± 0.001 | 0.395 ± 0.001 | 0.635 ± 0.010 | 0.703 ± 0.080 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.527 ± 0.015 | 1.553 ± 0.062 | 2.255 ± 0.209 | 2.999 ± 0.177 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.112 ± 0.001 | 0.111 ± 0.001 | 0.178 ± 0.005 | 0.108 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.036 ± 0.000 | 0.051 ± 0.001 | 0.071 ± 0.018 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.048 ± 0.004 | 0.052 ± 0.003 | 0.095 ± 0.002 | 0.154 ± 0.062 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.081 ± 0.000 | 0.065 ± 0.001 | 0.132 ± 0.019 | 0.192 ± 0.001 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.562 ± 0.004 | 0.557 ± 0.003 | 0.829 ± 0.011 | 0.848 ± 0.537 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.013 ± 0.000 | 0.173 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.178 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.189 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.259 ± 0.015 | 0.043 ± 0.000 | 0.194 ± 0.007 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.269 ± 0.005 | 0.049 ± 0.001 | 0.201 ± 0.113 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.283 ± 0.002 | 0.059 ± 0.001 | 0.206 ± 0.002 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.105 ± 0.001 | 0.043 ± 0.000 | 0.193 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.112 ± 0.002 | 0.049 ± 0.001 | 0.199 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.117 ± 0.002 | 0.059 ± 0.001 | 0.203 ± 0.003 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.745 ± 0.028 | 0.044 ± 0.000 | 0.385 ± 0.008 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.755 ± 0.010 | 0.045 ± 0.000 | 1.325 ± 0.586 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.749 ± 0.013 | 0.048 ± 0.000 | 0.378 ± 0.003 | 0.020 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.204 ± 0.002 | 0.041 ± 0.000 | 0.338 ± 0.010 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.204 ± 0.002 | 0.042 ± 0.011 | 0.337 ± 1.554 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.210 ± 0.003 | 0.045 ± 0.001 | 0.332 ± 0.009 | 0.020 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.080 ± 0.002 | 0.014 ± 0.000 | 0.071 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.082 ± 0.017 | 0.014 ± 0.000 | 0.102 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.082 ± 0.003 | 0.015 ± 0.000 | 0.074 ± 0.001 | 0.014 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.854 ± 0.011 | 0.067 ± 0.000 | 0.287 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.880 ± 0.010 | 0.069 ± 0.000 | 0.862 ± 0.169 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.896 ± 0.014 | 0.075 ± 0.001 | 0.292 ± 0.004 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.198 ± 0.002 | 0.070 ± 0.001 | 0.317 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.199 ± 0.002 | 0.072 ± 0.001 | 1.266 ± 0.398 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.205 ± 0.001 | 0.077 ± 0.000 | 0.317 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.129 ± 0.016 | 0.052 ± 0.000 | 0.119 ± 0.002 | 0.017 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.128 ± 0.001 | 0.052 ± 0.000 | 0.165 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.132 ± 0.005 | 0.056 ± 0.000 | 0.124 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.200 ± 0.003 | 0.036 ± 0.000 | 0.407 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.200 ± 0.003 | 0.036 ± 0.000 | 0.408 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.207 ± 0.002 | 0.037 ± 0.001 | 0.407 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.107 ± 0.002 | 0.036 ± 0.000 | 0.243 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.110 ± 0.001 | 0.036 ± 0.001 | 0.244 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.112 ± 0.001 | 0.039 ± 0.003 | 0.242 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.107 ± 0.004 | 0.026 ± 0.000 | 0.108 ± 0.002 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.115 ± 0.009 | 0.031 ± 0.000 | 0.148 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.121 ± 0.012 | 0.042 ± 0.000 | 0.127 ± 0.001 | 0.031 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.225 ± 0.003 | 0.027 ± 0.000 | 0.264 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.234 ± 0.002 | 0.031 ± 0.000 | 0.370 ± 0.008 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.251 ± 0.006 | 0.043 ± 0.001 | 0.286 ± 0.005 | 0.032 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.094 ± 0.002 | 0.026 ± 0.001 | 0.215 ± 0.038 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.098 ± 0.002 | 0.030 ± 0.000 | 0.219 ± 0.012 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.108 ± 0.002 | 0.042 ± 0.000 | 0.231 ± 0.003 | 0.032 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.032 ± 0.000 | 0.032 ± 0.000 | 0.034 ± 0.003 | 0.036 ± 0.009 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.028 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.028 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.001 | 0.016 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 40.5x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 41.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 43.4x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 56.3x faster than the slowest successful cell (`tenferro-eager`, 0.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 90.2x faster than the slowest successful cell (`pytorch-cpu`, 1.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 29.4x faster than the slowest successful cell (`pytorch-cpu`, 1.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 37.0x faster than the slowest successful cell (`tenferro-eager`, 0.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.749 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.338 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 64.5x faster than the slowest successful cell (`tenferro-eager`, 0.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 56.2x faster than the slowest successful cell (`tenferro-eager`, 0.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.880 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 41.1x faster than the slowest successful cell (`tenferro-eager`, 0.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.896 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 81.2x faster than the slowest successful cell (`pytorch-cpu`, 1.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 1.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.5x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 28.4x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.264 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

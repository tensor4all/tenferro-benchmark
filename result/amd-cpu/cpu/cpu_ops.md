# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `d9f6406557a0d1066f37785ec85f9b9afdbb4781`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261123_175252/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_175252/run.yaml`
- Timestamp: `2026-11-23T17:52:52.587828+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261123_175252`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261123_175252/cpu_ops_t1_20261123_175252.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.033 ± 0.000 | 0.032 ± 0.000 | 0.038 ± 0.000 | 0.031 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.038 ± 0.000 | 0.037 ± 0.000 | 0.039 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.137 ± 0.001 | 0.136 ± 0.001 | 0.132 ± 0.000 | 0.135 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.135 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.191 ± 0.012 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.135 ± 0.001 | 0.136 ± 0.000 | 0.163 ± 0.003 | 0.167 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.001 | 0.154 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.024 ± 0.001 | 0.157 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.049 ± 0.000 | 0.048 ± 0.000 | 0.041 ± 0.000 | 0.067 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.002 | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.023 ± 0.000 | 0.032 ± 0.011 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.023 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.000 | 0.041 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.027 ± 0.000 | 0.031 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.066 ± 0.000 | 0.065 ± 0.000 | 0.079 ± 0.002 | 0.086 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.058 ± 0.001 | 0.057 ± 0.001 | 0.075 ± 0.000 | 0.073 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.215 ± 0.001 | 0.214 ± 0.001 | 0.267 ± 0.001 | 0.238 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.066 ± 0.001 | 0.019 ± 0.000 | 0.114 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.087 ± 0.001 | 0.039 ± 0.004 | 0.119 ± 0.001 | 0.016 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.069 ± 0.010 | 0.019 ± 0.000 | 0.121 ± 0.003 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.091 ± 0.002 | 0.039 ± 0.000 | 0.147 ± 0.004 | 0.042 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.112 ± 0.007 | 0.048 ± 0.000 | 0.102 ± 0.001 | 0.023 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.128 ± 0.018 | 0.077 ± 0.000 | 0.112 ± 0.001 | 0.039 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.114 ± 0.004 | 0.054 ± 0.000 | 0.104 ± 0.003 | 0.026 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.137 ± 0.002 | 0.099 ± 0.001 | 0.124 ± 0.003 | 0.055 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.368 ± 0.001 | 0.368 ± 0.004 | 0.329 ± 0.007 | 0.408 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 25.408 ± 0.197 | 12.968 ± 0.061 | 11.082 ± 0.061 | 9.482 ± 0.025 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 132.043 ± 0.577 | 66.707 ± 0.076 | 71.739 ± 1.248 | 59.246 ± 0.161 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 17.467 ± 0.012 | 12.996 ± 0.020 | 9.069 ± 0.191 | 9.436 ± 0.014 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 94.202 ± 2.250 | 66.782 ± 0.581 | 55.873 ± 0.278 | 58.965 ± 0.283 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 11.226 ± 0.042 | 5.237 ± 0.019 | 13.256 ± 0.054 | 4.383 ± 0.026 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 80.719 ± 0.377 | 36.260 ± 0.409 | 87.316 ± 1.076 | 31.484 ± 0.284 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 10.271 ± 0.032 | 5.433 ± 0.029 | 5.845 ± 0.117 | 4.460 ± 0.021 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 78.036 ± 0.314 | 39.789 ± 2.324 | 46.338 ± 0.972 | 31.743 ± 0.679 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.000 | 0.020 ± 0.000 | 0.036 ± 0.002 | 0.043 ± 0.013 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.126 ± 0.025 | 0.056 ± 0.000 | 0.129 ± 0.003 | 0.082 ± 0.010 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 17.787 ± 0.138 | 8.793 ± 0.021 | 8.061 ± 0.029 | 6.954 ± 0.050 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 128.193 ± 2.774 | 61.005 ± 0.359 | 57.500 ± 0.260 | 48.698 ± 2.681 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 14.384 ± 0.027 | 8.696 ± 0.043 | 7.942 ± 0.096 | 7.168 ± 0.066 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 109.087 ± 1.710 | 60.057 ± 0.834 | 57.340 ± 0.544 | 52.068 ± 0.108 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.144 ± 0.009 | 0.112 ± 0.000 | 0.146 ± 0.003 | 0.074 ± 0.009 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 2.224 ± 0.007 | 1.118 ± 0.011 | 1.296 ± 0.019 | 0.686 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 11.659 ± 0.096 | 6.731 ± 0.068 | 5.962 ± 0.039 | 4.386 ± 0.022 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.281 ± 0.007 | 1.127 ± 0.001 | 1.807 ± 0.021 | 0.688 ± 0.003 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 7.074 ± 0.061 | 6.555 ± 0.037 | 10.189 ± 0.076 | 4.303 ± 0.041 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.883 ± 0.013 | 1.003 ± 0.005 | 0.805 ± 0.009 | 0.688 ± 0.076 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 45.611 ± 0.150 | 22.668 ± 0.143 | 22.479 ± 0.036 | 16.430 ± 0.203 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 278.138 ± 0.332 | 139.289 ± 0.474 | 149.708 ± 0.698 | 115.710 ± 0.352 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 30.764 ± 0.138 | 22.672 ± 0.093 | 17.048 ± 0.213 | 16.308 ± 0.103 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 186.972 ± 0.218 | 137.223 ± 0.667 | 119.788 ± 0.428 | 122.908 ± 0.794 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.107 ± 0.004 | 0.102 ± 0.001 | 0.228 ± 0.004 | 0.192 ± 0.008 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.814 ± 0.006 | 0.800 ± 0.009 | 1.248 ± 0.019 | 1.201 ± 0.016 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.165 ± 0.007 | 3.181 ± 0.021 | 4.070 ± 0.073 | 4.358 ± 0.017 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.112 ± 0.000 | 0.110 ± 0.000 | 0.113 ± 0.001 | 0.117 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.029 ± 0.000 | 0.048 ± 0.001 | 0.058 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.037 ± 0.000 | 0.034 ± 0.002 | 0.066 ± 0.004 | 0.070 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.069 ± 0.000 | 0.053 ± 0.000 | 0.124 ± 0.011 | 0.162 ± 0.001 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.691 ± 0.002 | 0.690 ± 0.002 | 0.649 ± 0.013 | 0.638 ± 0.003 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.133 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.133 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.001 | 0.013 ± 0.000 | 0.152 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.260 ± 0.003 | 0.026 ± 0.000 | 0.156 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.271 ± 0.005 | 0.029 ± 0.000 | 0.164 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.292 ± 0.003 | 0.039 ± 0.001 | 0.170 ± 0.004 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.088 ± 0.001 | 0.025 ± 0.000 | 0.148 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.096 ± 0.001 | 0.029 ± 0.001 | 0.157 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.108 ± 0.002 | 0.038 ± 0.001 | 0.161 ± 0.003 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.755 ± 0.004 | 0.033 ± 0.000 | 0.236 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.767 ± 0.007 | 0.033 ± 0.000 | 0.231 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.761 ± 0.004 | 0.035 ± 0.000 | 0.237 ± 0.004 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.187 ± 0.001 | 0.030 ± 0.000 | 0.215 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.189 ± 0.001 | 0.030 ± 0.000 | 0.217 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.197 ± 0.003 | 0.032 ± 0.000 | 0.222 ± 0.003 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.055 ± 0.002 | 0.012 ± 0.000 | 0.046 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.056 ± 0.000 | 0.012 ± 0.000 | 0.048 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.057 ± 0.005 | 0.012 ± 0.000 | 0.049 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 1.006 ± 0.003 | 0.043 ± 0.000 | 0.179 ± 0.006 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 1.020 ± 0.005 | 0.043 ± 0.000 | 0.178 ± 0.006 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 1.025 ± 0.006 | 0.048 ± 0.001 | 0.183 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.182 ± 0.003 | 0.043 ± 0.000 | 0.205 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.182 ± 0.004 | 0.044 ± 0.000 | 0.205 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.190 ± 0.002 | 0.048 ± 0.001 | 0.211 ± 0.006 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.101 ± 0.004 | 0.037 ± 0.000 | 0.084 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.101 ± 0.009 | 0.037 ± 0.001 | 0.087 ± 0.000 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.105 ± 0.003 | 0.039 ± 0.000 | 0.088 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.207 ± 0.001 | 0.026 ± 0.000 | 0.356 ± 0.010 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.209 ± 0.004 | 0.026 ± 0.000 | 0.360 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.212 ± 0.001 | 0.027 ± 0.000 | 0.357 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.100 ± 0.002 | 0.025 ± 0.000 | 0.198 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.102 ± 0.001 | 0.025 ± 0.000 | 0.199 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.105 ± 0.001 | 0.027 ± 0.001 | 0.199 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.079 ± 0.002 | 0.018 ± 0.000 | 0.080 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.083 ± 0.002 | 0.021 ± 0.000 | 0.089 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.095 ± 0.010 | 0.034 ± 0.000 | 0.100 ± 0.001 | 0.031 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.227 ± 0.001 | 0.019 ± 0.000 | 0.231 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.237 ± 0.001 | 0.023 ± 0.000 | 0.240 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.267 ± 0.003 | 0.035 ± 0.000 | 0.258 ± 0.008 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.084 ± 0.002 | 0.018 ± 0.000 | 0.174 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.089 ± 0.005 | 0.022 ± 0.000 | 0.181 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.107 ± 0.028 | 0.034 ± 0.000 | 0.194 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.033 ± 0.001 | 0.038 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.000 | 0.023 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.6x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`jax-cpu`, 0.135 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.2x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.7x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 33.9x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 33.6x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 35.1x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 25.7x faster than the slowest successful cell (`tenferro-eager`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.292 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 67.4x faster than the slowest successful cell (`tenferro-eager`, 0.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`tenferro-eager`, 0.755 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 66.2x faster than the slowest successful cell (`tenferro-eager`, 0.767 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 0.767 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 40.9x faster than the slowest successful cell (`tenferro-eager`, 0.761 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`tenferro-eager`, 0.761 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 90.7x faster than the slowest successful cell (`tenferro-eager`, 1.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 1.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 85.9x faster than the slowest successful cell (`tenferro-eager`, 1.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 23.5x faster than the slowest successful cell (`tenferro-eager`, 1.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 57.6x faster than the slowest successful cell (`tenferro-eager`, 1.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`tenferro-eager`, 1.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.211 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 32.0x faster than the slowest successful cell (`pytorch-cpu`, 0.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.360 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 29.5x faster than the slowest successful cell (`pytorch-cpu`, 0.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.357 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.240 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.240 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.267 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.181 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261123_183450/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261123_183450/run.yaml`
- Timestamp: `2026-11-23T18:34:50.691833+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261123_183450`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261123_183450/cpu_ops_t4_20261123_183450.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.033 ± 0.001 | 0.033 ± 0.000 | 0.036 ± 0.001 | 0.044 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.038 ± 0.000 | 0.038 ± 0.000 | 0.039 ± 0.000 | 0.041 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.138 ± 0.002 | 0.138 ± 0.000 | 0.130 ± 0.001 | 0.152 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.137 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.197 ± 0.016 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.137 ± 0.005 | 0.137 ± 0.001 | 0.115 ± 0.004 | 0.315 ± 0.086 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.159 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.000 | 0.160 ± 0.053 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.059 | 0.014 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.002 | 0.027 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.051 ± 0.000 | 0.053 ± 0.000 | 0.059 ± 0.000 | 0.071 ± 0.026 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.028 ± 0.002 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.034 ± 0.000 | 0.032 ± 0.007 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.024 ± 0.000 | 0.025 ± 0.000 | 0.037 ± 0.001 | 0.048 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.025 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.067 ± 0.000 | 0.066 ± 0.000 | 0.074 ± 0.001 | 0.086 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.059 ± 0.000 | 0.058 ± 0.000 | 0.073 ± 0.001 | 0.072 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.216 ± 0.000 | 0.216 ± 0.002 | 0.262 ± 0.001 | 0.254 ± 0.027 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.074 ± 0.001 | 0.022 ± 0.000 | 0.112 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.095 ± 0.006 | 0.042 ± 0.000 | 0.118 ± 0.002 | 0.015 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.076 ± 0.004 | 0.022 ± 0.000 | 0.119 ± 0.001 | 0.020 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.099 ± 0.002 | 0.043 ± 0.003 | 0.143 ± 0.002 | 0.076 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.126 ± 0.002 | 0.063 ± 0.000 | 0.134 ± 0.016 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.145 ± 0.018 | 0.092 ± 0.001 | 0.143 ± 0.001 | 0.058 ± 0.019 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.128 ± 0.004 | 0.069 ± 0.000 | 0.131 ± 0.002 | 0.028 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.153 ± 0.008 | 0.116 ± 0.001 | 0.173 ± 0.005 | 0.087 ± 0.010 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.372 ± 0.002 | 0.371 ± 0.003 | 0.349 ± 0.036 | 0.502 ± 0.009 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 26.864 ± 0.115 | 15.030 ± 0.171 | 7.829 ± 0.236 | 11.348 ± 1.492 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 134.371 ± 0.947 | 67.603 ± 0.195 | 44.634 ± 1.271 | 59.131 ± 3.493 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 19.648 ± 0.785 | 15.041 ± 0.039 | 6.513 ± 0.208 | 9.029 ± 0.807 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 91.307 ± 1.456 | 67.447 ± 0.072 | 32.643 ± 11.369 | 59.922 ± 6.762 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 10.242 ± 0.152 | 4.175 ± 0.042 | 8.306 ± 0.320 | 5.245 ± 1.869 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 49.323 ± 0.112 | 23.011 ± 0.067 | 49.249 ± 0.353 | 31.529 ± 12.717 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 8.842 ± 0.420 | 4.231 ± 0.007 | 3.890 ± 0.019 | 4.280 ± 0.923 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 44.559 ± 0.095 | 24.705 ± 1.044 | 25.669 ± 0.421 | 32.405 ± 4.471 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.037 ± 0.003 | 0.025 ± 0.000 | 0.034 ± 0.001 | 0.041 ± 0.010 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.158 ± 0.010 | 0.066 ± 0.001 | 0.148 ± 0.006 | 0.095 ± 0.011 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 17.083 ± 0.107 | 9.093 ± 0.013 | 5.151 ± 0.080 | 9.122 ± 2.614 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 72.881 ± 5.956 | 34.564 ± 0.813 | 35.611 ± 0.333 | 60.773 ± 10.622 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 12.515 ± 0.094 | 9.305 ± 0.033 | 5.713 ± 0.199 | 9.504 ± 1.243 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 58.988 ± 0.841 | 35.639 ± 0.620 | 34.394 ± 0.711 | 54.961 ± 7.163 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.197 ± 0.002 | 0.128 ± 0.001 | 0.145 ± 0.002 | 0.089 ± 0.029 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 2.959 ± 0.517 | 1.185 ± 0.019 | 1.041 ± 0.036 | 0.927 ± 0.083 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 9.300 ± 0.226 | 5.070 ± 0.033 | 3.124 ± 0.038 | 4.052 ± 0.113 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.856 ± 0.140 | 1.229 ± 0.031 | 1.292 ± 0.012 | 0.953 ± 0.177 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 5.592 ± 0.250 | 5.210 ± 0.028 | 5.049 ± 0.078 | 3.877 ± 0.209 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.042 ± 0.006 | 1.219 ± 0.004 | 1.051 ± 0.022 | 0.899 ± 0.014 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 47.075 ± 0.055 | 24.295 ± 0.084 | 17.877 ± 0.108 | 17.269 ± 1.236 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 280.426 ± 0.864 | 135.353 ± 0.359 | 100.667 ± 15.142 | 107.955 ± 3.010 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 32.238 ± 0.520 | 24.297 ± 0.025 | 16.406 ± 0.463 | 17.134 ± 1.592 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 187.035 ± 0.366 | 134.984 ± 1.406 | 89.841 ± 1.556 | 121.022 ± 10.560 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.067 ± 0.001 | 0.067 ± 0.000 | 0.147 ± 0.001 | 0.152 ± 0.047 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.427 ± 0.002 | 0.426 ± 0.006 | 0.641 ± 0.034 | 0.664 ± 0.096 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.633 ± 0.003 | 1.631 ± 0.004 | 2.067 ± 0.047 | 2.244 ± 0.108 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.114 ± 0.002 | 0.113 ± 0.001 | 0.188 ± 0.019 | 0.113 ± 0.015 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.029 ± 0.000 | 0.032 ± 0.000 | 0.048 ± 0.001 | 0.075 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.038 ± 0.004 | 0.038 ± 0.001 | 0.093 ± 0.001 | 0.147 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.072 ± 0.000 | 0.057 ± 0.006 | 0.131 ± 0.005 | 0.196 ± 0.018 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.700 ± 0.011 | 0.703 ± 0.018 | 0.916 ± 0.033 | 0.837 ± 0.014 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.018 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.134 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.136 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.150 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.277 ± 0.002 | 0.037 ± 0.000 | 0.155 ± 0.006 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.288 ± 0.002 | 0.041 ± 0.000 | 0.161 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.305 ± 0.002 | 0.050 ± 0.000 | 0.171 ± 0.003 | 0.017 ± 0.007 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.106 ± 0.002 | 0.035 ± 0.000 | 0.146 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.116 ± 0.003 | 0.038 ± 0.000 | 0.154 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.127 ± 0.002 | 0.048 ± 0.001 | 0.162 ± 0.005 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.791 ± 0.022 | 0.047 ± 0.000 | 0.325 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.795 ± 0.009 | 0.048 ± 0.000 | 0.323 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.807 ± 0.002 | 0.050 ± 0.000 | 0.322 ± 0.005 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.235 ± 0.004 | 0.042 ± 0.000 | 0.284 ± 0.048 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.237 ± 0.005 | 0.042 ± 0.011 | 0.269 ± 0.006 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.243 ± 0.002 | 0.045 ± 0.000 | 0.271 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.064 ± 0.001 | 0.016 ± 0.004 | 0.046 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.066 ± 0.011 | 0.016 ± 0.000 | 0.048 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.065 ± 0.002 | 0.016 ± 0.000 | 0.047 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 1.029 ± 0.004 | 0.061 ± 0.000 | 0.234 ± 0.006 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 1.060 ± 0.023 | 0.063 ± 0.001 | 0.230 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 1.062 ± 0.012 | 0.067 ± 0.001 | 0.237 ± 0.007 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.227 ± 0.001 | 0.060 ± 0.001 | 0.257 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.228 ± 0.001 | 0.061 ± 0.001 | 0.253 ± 0.005 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.240 ± 0.006 | 0.066 ± 0.014 | 0.255 ± 0.006 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.117 ± 0.007 | 0.051 ± 0.000 | 0.085 ± 0.002 | 0.017 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.115 ± 0.005 | 0.052 ± 0.000 | 0.088 ± 0.000 | 0.015 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.118 ± 0.005 | 0.054 ± 0.001 | 0.090 ± 0.003 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.219 ± 0.001 | 0.036 ± 0.000 | 0.359 ± 0.014 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.222 ± 0.001 | 0.036 ± 0.000 | 0.355 ± 0.011 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.228 ± 0.001 | 0.038 ± 0.000 | 0.356 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.117 ± 0.001 | 0.035 ± 0.000 | 0.194 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.116 ± 0.001 | 0.036 ± 0.000 | 0.197 ± 0.007 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.120 ± 0.001 | 0.037 ± 0.000 | 0.197 ± 0.005 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.092 ± 0.013 | 0.023 ± 0.000 | 0.077 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.096 ± 0.009 | 0.027 ± 0.000 | 0.087 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.110 ± 0.010 | 0.039 ± 0.000 | 0.099 ± 0.001 | 0.023 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.237 ± 0.001 | 0.025 ± 0.000 | 0.233 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.248 ± 0.002 | 0.029 ± 0.000 | 0.242 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.277 ± 0.002 | 0.042 ± 0.001 | 0.260 ± 0.010 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.096 ± 0.003 | 0.023 ± 0.000 | 0.170 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.102 ± 0.002 | 0.027 ± 0.000 | 0.179 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.120 ± 0.002 | 0.040 ± 0.000 | 0.192 ± 0.004 | 0.025 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.036 ± 0.002 | 0.039 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.000 | 0.018 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.000 | 0.023 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 27.9x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 27.8x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 27.7x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.9x faster than the slowest successful cell (`tenferro-eager`, 0.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.305 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.3x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 70.1x faster than the slowest successful cell (`tenferro-eager`, 0.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.791 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.8x faster than the slowest successful cell (`tenferro-eager`, 0.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 0.795 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 38.1x faster than the slowest successful cell (`tenferro-eager`, 0.807 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.2x faster than the slowest successful cell (`tenferro-eager`, 0.807 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.3x faster than the slowest successful cell (`pytorch-cpu`, 0.284 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 92.5x faster than the slowest successful cell (`tenferro-eager`, 1.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 1.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 81.5x faster than the slowest successful cell (`tenferro-eager`, 1.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 1.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 52.1x faster than the slowest successful cell (`tenferro-eager`, 1.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 1.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 32.4x faster than the slowest successful cell (`pytorch-cpu`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 0.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.3x faster than the slowest successful cell (`pytorch-cpu`, 0.356 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.3x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `9185e8ca26c3fc600458361d0f477495a717daf3`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261124_175224/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_175224/run.yaml`
- Timestamp: `2026-11-24T17:52:24.640647+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261124_175224`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261124_175224/cpu_ops_t1_20261124_175224.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.028 ± 0.000 | 0.028 ± 0.001 | 0.033 ± 0.002 | 0.037 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.032 ± 0.000 | 0.032 ± 0.000 | 0.034 ± 0.001 | 0.037 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.118 ± 0.004 | 0.118 ± 0.001 | 0.115 ± 0.005 | 0.115 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.124 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.162 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.120 ± 0.001 | 0.120 ± 0.001 | 0.141 ± 0.002 | 0.184 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.165 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.171 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.042 ± 0.000 | 0.042 ± 0.000 | 0.035 ± 0.001 | 0.060 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.014 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.024 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.018 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.056 ± 0.000 | 0.057 ± 0.001 | 0.070 ± 0.003 | 0.077 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.001 | 0.050 ± 0.000 | 0.064 ± 0.001 | 0.064 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.186 ± 0.002 | 0.188 ± 0.002 | 0.229 ± 0.004 | 0.222 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.057 ± 0.002 | 0.017 ± 0.000 | 0.106 ± 0.003 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.077 ± 0.005 | 0.034 ± 0.000 | 0.110 ± 0.004 | 0.018 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.057 ± 0.002 | 0.018 ± 0.000 | 0.109 ± 0.002 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.077 ± 0.001 | 0.035 ± 0.000 | 0.129 ± 0.003 | 0.031 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.097 ± 0.001 | 0.045 ± 0.000 | 0.090 ± 0.002 | 0.022 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.116 ± 0.008 | 0.070 ± 0.001 | 0.096 ± 0.004 | 0.037 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.099 ± 0.010 | 0.049 ± 0.000 | 0.096 ± 0.002 | 0.022 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.116 ± 0.001 | 0.090 ± 0.001 | 0.107 ± 0.003 | 0.061 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.293 ± 0.003 | 0.295 ± 0.005 | 0.281 ± 0.004 | 0.330 ± 0.008 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 15.406 ± 0.070 | 11.195 ± 0.238 | 9.907 ± 0.147 | 7.688 ± 0.070 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 80.615 ± 0.628 | 63.265 ± 0.709 | 65.704 ± 0.386 | 49.623 ± 0.670 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 8.292 ± 0.312 | 11.173 ± 0.169 | 8.321 ± 0.131 | 7.860 ± 0.148 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 47.212 ± 0.588 | 62.515 ± 0.697 | 51.857 ± 0.370 | 50.047 ± 1.235 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 9.307 ± 0.220 | 4.346 ± 0.086 | 12.378 ± 0.283 | 4.660 ± 0.122 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 60.730 ± 1.022 | 29.940 ± 0.263 | 78.713 ± 1.411 | 30.270 ± 0.291 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 8.264 ± 0.036 | 4.562 ± 0.102 | 5.302 ± 0.160 | 4.377 ± 0.032 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 57.499 ± 0.302 | 31.941 ± 0.345 | 42.740 ± 1.269 | 30.258 ± 0.913 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.000 | 0.017 ± 0.000 | 0.034 ± 0.001 | 0.045 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.107 ± 0.001 | 0.048 ± 0.000 | 0.115 ± 0.001 | 0.089 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 12.921 ± 0.157 | 7.293 ± 0.103 | 7.431 ± 0.384 | 6.245 ± 0.081 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 85.779 ± 1.265 | 49.284 ± 0.138 | 51.906 ± 1.080 | 41.626 ± 0.224 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 9.971 ± 0.133 | 7.203 ± 0.044 | 7.081 ± 0.164 | 6.030 ± 0.057 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 71.252 ± 1.996 | 48.893 ± 0.386 | 51.349 ± 2.711 | 46.079 ± 0.944 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.131 ± 0.004 | 0.104 ± 0.001 | 0.128 ± 0.004 | 0.054 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.179 ± 0.022 | 0.993 ± 0.012 | 1.110 ± 0.021 | 0.629 ± 0.006 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 5.939 ± 0.099 | 5.694 ± 0.078 | 5.552 ± 0.250 | 3.936 ± 0.102 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.651 ± 0.007 | 0.997 ± 0.007 | 1.583 ± 0.032 | 0.633 ± 0.010 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.020 ± 0.079 | 5.519 ± 0.063 | 9.241 ± 0.820 | 3.907 ± 0.062 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.672 ± 0.015 | 0.823 ± 0.012 | 0.738 ± 0.031 | 0.542 ± 0.004 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 27.167 ± 0.073 | 20.196 ± 0.358 | 20.050 ± 0.052 | 13.808 ± 0.115 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 153.664 ± 2.162 | 115.866 ± 1.776 | 135.070 ± 1.196 | 97.333 ± 1.702 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.203 ± 0.078 | 20.099 ± 0.221 | 15.558 ± 0.919 | 13.478 ± 0.791 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 82.414 ± 1.047 | 114.690 ± 1.860 | 107.584 ± 0.971 | 103.130 ± 0.683 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.109 ± 0.002 | 0.089 ± 0.000 | 0.200 ± 0.003 | 0.171 ± 0.006 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.740 ± 0.012 | 0.672 ± 0.001 | 1.045 ± 0.011 | 1.030 ± 0.024 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 2.793 ± 0.035 | 2.684 ± 0.014 | 3.463 ± 0.138 | 3.667 ± 0.281 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.097 ± 0.001 | 0.094 ± 0.001 | 0.103 ± 0.002 | 0.095 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.042 ± 0.000 | 0.040 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.031 ± 0.000 | 0.029 ± 0.001 | 0.065 ± 0.002 | 0.066 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.057 ± 0.000 | 0.045 ± 0.001 | 0.111 ± 0.007 | 0.139 ± 0.005 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.565 ± 0.013 | 0.547 ± 0.039 | 0.563 ± 0.012 | 0.493 ± 0.016 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.122 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.117 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.136 ± 0.005 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.213 ± 0.000 | 0.024 ± 0.000 | 0.138 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.219 ± 0.004 | 0.026 ± 0.000 | 0.140 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.233 ± 0.006 | 0.035 ± 0.000 | 0.143 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.065 ± 0.002 | 0.022 ± 0.001 | 0.128 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.070 ± 0.001 | 0.025 ± 0.000 | 0.134 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.075 ± 0.001 | 0.034 ± 0.001 | 0.135 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.628 ± 0.017 | 0.029 ± 0.000 | 0.203 ± 0.009 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.634 ± 0.010 | 0.029 ± 0.000 | 0.198 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.627 ± 0.006 | 0.032 ± 0.001 | 0.205 ± 0.007 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.140 ± 0.002 | 0.026 ± 0.000 | 0.183 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.146 ± 0.001 | 0.026 ± 0.000 | 0.184 ± 0.007 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.147 ± 0.001 | 0.030 ± 0.000 | 0.189 ± 0.005 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.058 ± 0.038 | 0.011 ± 0.000 | 0.040 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.050 ± 0.001 | 0.011 ± 0.000 | 0.043 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.054 ± 0.007 | 0.011 ± 0.000 | 0.044 ± 0.002 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.828 ± 0.007 | 0.037 ± 0.000 | 0.156 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.870 ± 0.016 | 0.039 ± 0.000 | 0.156 ± 0.010 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.855 ± 0.010 | 0.044 ± 0.001 | 0.154 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.139 ± 0.004 | 0.037 ± 0.001 | 0.178 ± 0.005 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.142 ± 0.001 | 0.039 ± 0.000 | 0.178 ± 0.005 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.146 ± 0.004 | 0.044 ± 0.000 | 0.182 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.089 ± 0.004 | 0.032 ± 0.000 | 0.073 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.090 ± 0.001 | 0.032 ± 0.000 | 0.075 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.095 ± 0.001 | 0.035 ± 0.000 | 0.075 ± 0.009 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.157 ± 0.002 | 0.023 ± 0.000 | 0.300 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.159 ± 0.003 | 0.023 ± 0.000 | 0.306 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.165 ± 0.002 | 0.025 ± 0.000 | 0.307 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.069 ± 0.000 | 0.023 ± 0.000 | 0.166 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.071 ± 0.002 | 0.023 ± 0.000 | 0.168 ± 0.007 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.075 ± 0.001 | 0.024 ± 0.000 | 0.170 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.069 ± 0.002 | 0.016 ± 0.000 | 0.069 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.073 ± 0.010 | 0.019 ± 0.000 | 0.076 ± 0.002 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.085 ± 0.010 | 0.030 ± 0.000 | 0.087 ± 0.002 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.177 ± 0.006 | 0.016 ± 0.000 | 0.200 ± 0.007 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.187 ± 0.003 | 0.020 ± 0.000 | 0.206 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.205 ± 0.005 | 0.031 ± 0.001 | 0.214 ± 0.003 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.058 ± 0.000 | 0.015 ± 0.000 | 0.150 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.062 ± 0.003 | 0.019 ± 0.000 | 0.163 ± 0.006 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.072 ± 0.002 | 0.031 ± 0.001 | 0.161 ± 0.006 | 0.017 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.020 ± 0.001 | 0.014 ± 0.000 | 0.031 ± 0.001 | 0.034 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.000 | 0.023 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`jax-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.7x faster than the slowest successful cell (`jax-cpu`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 27.0x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 26.0x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 30.3x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.9x faster than the slowest successful cell (`tenferro-eager`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 21.3x faster than the slowest successful cell (`tenferro-eager`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 62.3x faster than the slowest successful cell (`tenferro-eager`, 0.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 0.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 61.3x faster than the slowest successful cell (`tenferro-eager`, 0.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 0.634 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 40.8x faster than the slowest successful cell (`tenferro-eager`, 0.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 0.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 79.9x faster than the slowest successful cell (`tenferro-eager`, 0.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 0.828 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 79.9x faster than the slowest successful cell (`tenferro-eager`, 0.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 0.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 52.1x faster than the slowest successful cell (`tenferro-eager`, 0.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`tenferro-eager`, 0.855 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 28.7x faster than the slowest successful cell (`pytorch-cpu`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`pytorch-cpu`, 0.306 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 28.1x faster than the slowest successful cell (`pytorch-cpu`, 0.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.307 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261124_182830/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261124_182830/run.yaml`
- Timestamp: `2026-11-24T18:28:30.406333+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261124_182830`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261124_182830/cpu_ops_t4_20261124_182830.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.000 | 0.028 ± 0.005 | 0.032 ± 0.001 | 0.036 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.034 ± 0.000 | 0.033 ± 0.000 | 0.035 ± 0.001 | 0.038 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.120 ± 0.001 | 0.118 ± 0.002 | 0.113 ± 0.002 | 0.127 ± 0.016 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.120 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.129 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.058 ± 0.001 | 0.058 ± 0.017 | 0.099 ± 0.003 | 0.195 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.177 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.184 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.019 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.002 | 0.022 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.001 | 0.042 ± 0.000 | 0.051 ± 0.002 | 0.053 ± 0.021 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.025 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.031 ± 0.001 | 0.028 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.027 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.023 ± 0.001 | 0.023 ± 0.000 | 0.033 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.023 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.059 ± 0.001 | 0.057 ± 0.000 | 0.063 ± 0.002 | 0.079 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.052 ± 0.001 | 0.053 ± 0.000 | 0.065 ± 0.004 | 0.066 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.191 ± 0.003 | 0.188 ± 0.004 | 0.226 ± 0.005 | 0.215 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.071 ± 0.006 | 0.020 ± 0.000 | 0.099 ± 0.002 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.084 ± 0.001 | 0.037 ± 0.000 | 0.107 ± 0.002 | 0.013 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.070 ± 0.001 | 0.020 ± 0.000 | 0.104 ± 0.003 | 0.020 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.099 ± 0.014 | 0.038 ± 0.000 | 0.132 ± 0.006 | 0.064 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.113 ± 0.005 | 0.060 ± 0.001 | 0.113 ± 0.002 | 0.022 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.127 ± 0.007 | 0.083 ± 0.001 | 0.135 ± 0.006 | 0.045 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.118 ± 0.006 | 0.062 ± 0.001 | 0.116 ± 0.004 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.161 ± 0.011 | 0.105 ± 0.002 | 0.142 ± 0.004 | 0.071 ± 0.021 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.304 ± 0.002 | 0.297 ± 0.001 | 0.286 ± 0.001 | 0.340 ± 0.007 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 17.139 ± 0.087 | 13.027 ± 0.186 | 6.902 ± 0.437 | 9.955 ± 1.173 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 53.424 ± 1.017 | 42.798 ± 0.150 | 40.973 ± 1.096 | 46.288 ± 1.800 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 8.954 ± 0.123 | 12.969 ± 0.272 | 5.971 ± 0.142 | 9.632 ± 1.616 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 30.274 ± 0.203 | 42.860 ± 0.761 | 29.389 ± 0.468 | 50.127 ± 4.730 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.682 ± 0.198 | 3.494 ± 0.042 | 7.832 ± 0.172 | 6.338 ± 1.265 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 35.804 ± 0.426 | 18.865 ± 0.363 | 45.012 ± 0.593 | 32.297 ± 2.863 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 5.277 ± 0.101 | 3.580 ± 0.022 | 3.517 ± 0.070 | 6.504 ± 0.063 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 32.350 ± 0.305 | 19.785 ± 0.146 | 23.236 ± 1.032 | 29.864 ± 6.303 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.038 ± 0.001 | 0.023 ± 0.000 | 0.030 ± 0.000 | 0.037 ± 0.007 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.150 ± 0.008 | 0.061 ± 0.001 | 0.129 ± 0.004 | 0.066 ± 0.018 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 8.816 ± 0.243 | 4.937 ± 0.078 | 4.591 ± 0.064 | 8.097 ± 1.895 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 48.733 ± 1.607 | 28.846 ± 0.731 | 32.126 ± 1.110 | 52.026 ± 13.698 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 6.131 ± 0.120 | 4.969 ± 0.016 | 5.176 ± 0.216 | 9.604 ± 2.306 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 38.684 ± 0.727 | 29.470 ± 0.383 | 31.023 ± 0.508 | 52.464 ± 21.578 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.174 ± 0.008 | 0.159 ± 0.004 | 0.128 ± 0.003 | 0.072 ± 0.019 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.235 ± 0.108 | 0.954 ± 0.012 | 1.727 ± 1.535 | 0.804 ± 0.042 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 4.697 ± 0.092 | 4.122 ± 0.089 | 2.808 ± 0.024 | 3.366 ± 0.054 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.658 ± 0.011 | 1.006 ± 0.047 | 1.432 ± 0.066 | 0.901 ± 0.146 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 2.362 ± 0.046 | 4.134 ± 0.142 | 4.507 ± 0.138 | 3.539 ± 0.019 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.885 ± 0.011 | 1.029 ± 0.002 | 0.924 ± 0.021 | 0.770 ± 0.054 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 30.490 ± 0.275 | 21.128 ± 0.147 | 15.994 ± 0.328 | 15.891 ± 1.315 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 100.848 ± 2.253 | 79.919 ± 1.969 | 90.797 ± 5.254 | 97.875 ± 4.602 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 15.354 ± 0.082 | 21.092 ± 0.230 | 14.433 ± 0.173 | 16.972 ± 2.952 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 52.753 ± 0.565 | 81.127 ± 2.602 | 80.205 ± 3.685 | 95.060 ± 6.728 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.066 ± 0.003 | 0.061 ± 0.000 | 0.135 ± 0.001 | 0.152 ± 0.032 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.380 ± 0.003 | 0.364 ± 0.004 | 0.565 ± 0.013 | 0.653 ± 0.099 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.447 ± 0.020 | 1.395 ± 0.053 | 1.823 ± 0.064 | 2.156 ± 0.374 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.100 ± 0.001 | 0.094 ± 0.003 | 0.160 ± 0.005 | 0.101 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.042 ± 0.002 | 0.047 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.050 ± 0.002 | 0.045 ± 0.002 | 0.082 ± 0.003 | 0.130 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.098 ± 0.002 | 0.072 ± 0.001 | 0.122 ± 0.001 | 0.192 ± 0.027 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.565 ± 0.005 | 0.550 ± 0.003 | 0.761 ± 0.013 | 0.689 ± 0.044 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.119 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.116 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.125 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.225 ± 0.003 | 0.034 ± 0.001 | 0.134 ± 0.006 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.237 ± 0.011 | 0.038 ± 0.001 | 0.134 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.253 ± 0.001 | 0.045 ± 0.001 | 0.146 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.078 ± 0.001 | 0.032 ± 0.000 | 0.124 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.084 ± 0.001 | 0.036 ± 0.007 | 0.131 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.091 ± 0.001 | 0.043 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.661 ± 0.007 | 0.042 ± 0.000 | 0.277 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.678 ± 0.011 | 0.042 ± 0.000 | 0.363 ± 0.134 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.702 ± 0.004 | 0.045 ± 0.001 | 0.275 ± 0.008 | 0.019 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.180 ± 0.001 | 0.037 ± 0.001 | 0.231 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.184 ± 0.005 | 0.038 ± 0.000 | 0.233 ± 0.025 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.188 ± 0.001 | 0.040 ± 0.000 | 0.235 ± 0.004 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.056 ± 0.001 | 0.015 ± 0.000 | 0.039 ± 0.001 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.058 ± 0.002 | 0.015 ± 0.000 | 0.040 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.059 ± 0.005 | 0.015 ± 0.000 | 0.043 ± 0.001 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.887 ± 0.002 | 0.056 ± 0.002 | 0.202 ± 0.059 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.885 ± 0.011 | 0.059 ± 0.001 | 0.193 ± 0.007 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.898 ± 0.028 | 0.062 ± 0.001 | 0.202 ± 0.160 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.176 ± 0.025 | 0.055 ± 0.001 | 0.212 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.172 ± 0.003 | 0.057 ± 0.001 | 0.215 ± 0.009 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.185 ± 0.005 | 0.060 ± 0.000 | 0.222 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.098 ± 0.007 | 0.048 ± 0.000 | 0.072 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.105 ± 0.006 | 0.047 ± 0.001 | 0.073 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.105 ± 0.002 | 0.050 ± 0.008 | 0.075 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.169 ± 0.002 | 0.033 ± 0.000 | 0.298 ± 0.006 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.177 ± 0.005 | 0.033 ± 0.000 | 0.301 ± 0.007 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.183 ± 0.007 | 0.034 ± 0.000 | 0.298 ± 0.003 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.081 ± 0.004 | 0.032 ± 0.001 | 0.165 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.084 ± 0.002 | 0.032 ± 0.000 | 0.164 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.089 ± 0.001 | 0.033 ± 0.001 | 0.164 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.078 ± 0.007 | 0.021 ± 0.000 | 0.068 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.085 ± 0.004 | 0.025 ± 0.000 | 0.074 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.093 ± 0.001 | 0.036 ± 0.001 | 0.088 ± 0.002 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.186 ± 0.002 | 0.023 ± 0.000 | 0.197 ± 0.005 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.198 ± 0.003 | 0.027 ± 0.001 | 0.202 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.214 ± 0.001 | 0.038 ± 0.000 | 0.213 ± 0.003 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.069 ± 0.002 | 0.021 ± 0.001 | 0.144 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.073 ± 0.002 | 0.025 ± 0.000 | 0.150 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.077 ± 0.001 | 0.036 ± 0.000 | 0.162 ± 0.004 | 0.021 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 1.285 | 0.013 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.018 ± 0.001 | 0.023 ± 0.000 | 0.031 ± 0.000 | 0.045 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.018 ± 0.000 | 0.021 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 27.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 23.8x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 25.8x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 26.4x faster than the slowest successful cell (`tenferro-eager`, 0.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 22.4x faster than the slowest successful cell (`tenferro-eager`, 0.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 68.3x faster than the slowest successful cell (`tenferro-eager`, 0.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.661 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.1x faster than the slowest successful cell (`tenferro-eager`, 0.678 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.678 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 37.8x faster than the slowest successful cell (`tenferro-eager`, 0.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.702 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.231 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.233 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.235 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 91.4x faster than the slowest successful cell (`tenferro-eager`, 0.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 80.7x faster than the slowest successful cell (`tenferro-eager`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`tenferro-eager`, 0.885 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 50.8x faster than the slowest successful cell (`tenferro-eager`, 0.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.898 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 0.301 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 28.8x faster than the slowest successful cell (`pytorch-cpu`, 0.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.165 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-eager`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`
- Benchmark commit: `ce89bccec28bc7c863b30c80764d3570747f6df6`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261019_175954/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261019_175954`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261019_175954`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261019_175954/cpu_ops_t1_20261019_175954.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261019_175954/cpu_ops_t1_20261019_175954.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.022 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.025 ± 0.000 | 0.026 ± 0.000 | 0.025 ± 0.001 | 0.028 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.091 ± 0.000 | 0.090 ± 0.000 | 0.088 ± 0.000 | 0.084 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.077 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.098 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.069 ± 0.000 | 0.078 ± 0.000 | 0.081 ± 0.001 | 0.120 ± 0.027 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.096 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.087 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.018 ± 0.001 | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.021 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.057 ± 0.001 | 0.057 ± 0.000 | 0.048 ± 0.000 | 0.061 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.001 | 0.051 ± 0.000 | 0.045 ± 0.001 | 0.050 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.186 ± 0.001 | 0.186 ± 0.001 | 0.175 ± 0.001 | 0.160 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.047 ± 0.001 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.000 | 0.058 ± 0.001 | 0.012 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.051 ± 0.001 | 0.012 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.069 ± 0.001 | 0.032 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.039 ± 0.001 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.047 ± 0.001 | 0.028 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.038 ± 0.002 | 0.023 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.051 ± 0.001 | 0.039 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.183 ± 0.002 | 0.185 ± 0.001 | 0.193 ± 0.002 | 0.210 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 5.408 ± 0.060 | 6.724 ± 0.058 | 5.088 ± 0.030 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 35.429 ± 0.241 | 46.609 ± 0.401 | 33.723 ± 0.395 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 5.534 ± 0.068 | 5.780 ± 0.657 | 5.130 ± 0.066 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 35.328 ± 0.063 | 34.903 ± 0.369 | 33.113 ± 0.668 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 3.305 ± 0.084 | 9.204 ± 0.569 | 2.815 ± 0.039 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 24.712 ± 0.108 | 64.878 ± 1.578 | 22.447 ± 0.461 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 3.258 ± 0.016 | 4.221 ± 0.052 | 2.805 ± 0.026 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 23.670 ± 0.362 | 31.096 ± 0.397 | 21.705 ± 0.269 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.023 ± 0.001 | 0.018 ± 0.000 | 0.022 ± 0.001 | 0.022 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.055 ± 0.001 | 0.069 ± 0.002 | 0.046 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 4.473 ± 0.005 | 5.129 ± 0.125 | 3.890 ± 0.052 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 37.158 ± 0.246 | 44.451 ± 1.456 | 28.302 ± 0.271 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 4.515 ± 0.064 | 5.139 ± 0.048 | 3.991 ± 0.044 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 35.306 ± 0.257 | 43.816 ± 0.892 | 30.918 ± 0.176 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.061 ± 0.000 | 0.037 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.430 ± 0.000 | 0.650 ± 0.022 | 0.367 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.587 ± 0.008 | 4.002 ± 0.076 | 2.498 ± 0.029 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.422 ± 0.003 | 1.125 ± 0.181 | 0.381 ± 0.033 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.549 ± 0.005 | 6.137 ± 0.082 | 2.534 ± 0.027 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.374 ± 0.002 | 0.413 ± 0.005 | 0.353 ± 0.008 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 9.577 ± 0.044 | 14.537 ± 0.326 | 8.855 ± 0.040 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 66.527 ± 0.609 | 99.559 ± 2.664 | 65.897 ± 1.534 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 9.558 ± 0.030 | 10.275 ± 0.143 | 8.947 ± 0.050 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 67.064 ± 0.835 | 83.559 ± 1.385 | 68.633 ± 2.993 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.086 ± 0.005 | 0.094 ± 0.001 | 0.137 ± 0.002 | 0.103 ± 0.001 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.595 ± 0.004 | 0.609 ± 0.003 | 0.759 ± 0.023 | 0.758 ± 0.017 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 2.284 ± 0.096 | 2.339 ± 0.013 | 2.565 ± 0.117 | 2.875 ± 0.483 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.059 ± 0.001 | 0.058 ± 0.000 | 0.079 ± 0.001 | 0.063 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.021 ± 0.001 | 0.022 ± 0.000 | 0.029 ± 0.001 | 0.027 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.026 ± 0.001 | 0.028 ± 0.001 | 0.033 ± 0.001 | 0.036 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.044 ± 0.000 | 0.047 ± 0.001 | 0.056 ± 0.002 | 0.087 ± 0.015 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.337 ± 0.001 | 0.342 ± 0.003 | 0.360 ± 0.002 | 0.328 ± 0.004 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.009 ± 0.002 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.069 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.071 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.091 ± 0.014 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.019 ± 0.001 | 0.061 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.021 ± 0.000 | 0.061 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.000 | 0.064 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.019 ± 0.001 | 0.051 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.021 ± 0.000 | 0.051 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.023 ± 0.000 | 0.054 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.000 | 0.102 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.000 | 0.098 ± 0.001 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.024 ± 0.000 | 0.099 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.022 ± 0.000 | 0.076 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.023 ± 0.000 | 0.077 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.024 ± 0.000 | 0.078 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.020 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.020 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.034 ± 0.002 | 0.070 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.035 ± 0.000 | 0.068 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.036 ± 0.000 | 0.070 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.033 ± 0.000 | 0.070 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.035 ± 0.000 | 0.070 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.036 ± 0.000 | 0.070 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.012 | 0.032 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.034 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.033 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.013 ± 0.000 | 0.157 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.014 ± 0.000 | 0.157 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.014 ± 0.000 | 0.157 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.012 ± 0.000 | 0.067 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.013 ± 0.000 | 0.068 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.013 ± 0.002 | 0.068 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.024 ± 0.014 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.025 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.033 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.012 ± 0.000 | 0.094 ± 0.002 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.095 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.021 ± 0.000 | 0.105 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.012 ± 0.000 | 0.056 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.059 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.021 ± 0.000 | 0.065 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.015 ± 0.001 | 0.017 ± 0.001 | 0.028 ± 0.001 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.007 ± 0.003 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 31.8x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 31.0x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 20.7x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.095 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261019_182327/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Timestamp: `20261019_182327`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261019_182327`.

- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

### CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261019_182327/cpu_ops_t4_20261019_182327.csv`
- Source table: `data/results/amd-cpu/cpu/einsum/20261019_182327/cpu_ops_t4_20261019_182327.md`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.025 ± 0.000 | 0.025 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.062 ± 0.000 | 0.059 ± 0.001 | 0.020 ± 0.000 | 0.031 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.058 ± 0.001 | 0.055 ± 0.000 | 0.024 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.120 ± 0.003 | 0.120 ± 0.001 | 0.077 ± 0.000 | 0.093 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.027 ± 0.001 | 0.008 ± 0.000 | 0.069 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.026 ± 0.000 | 0.009 ± 0.000 | 0.072 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.116 ± 0.003 | 0.143 ± 0.001 | 0.128 ± 0.008 | 0.168 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.024 ± 0.000 | 0.009 ± 0.000 | 0.077 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.007 ± 0.000 | 0.027 ± 0.000 | 0.013 ± 0.000 | 0.074 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.019 ± 0.001 | 0.016 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.029 ± 0.001 | 0.028 ± 0.001 | 0.016 ± 0.001 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.022 ± 0.001 | 0.022 ± 0.001 | 0.013 ± 0.000 | 0.015 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.060 ± 0.001 | 0.056 ± 0.000 | 0.032 ± 0.001 | 0.041 ± 0.010 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.016 ± 0.001 | 0.010 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.030 ± 0.001 | 0.029 ± 0.002 | 0.018 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.026 ± 0.002 | 0.026 ± 0.003 | 0.016 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.033 ± 0.001 | 0.032 ± 0.001 | 0.021 ± 0.000 | 0.027 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.036 ± 0.007 | 0.034 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.086 ± 0.000 | 0.086 ± 0.000 | 0.043 ± 0.001 | 0.054 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.001 | 0.081 ± 0.000 | 0.042 ± 0.001 | 0.046 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.213 ± 0.001 | 0.209 ± 0.001 | 0.160 ± 0.006 | 0.150 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.087 ± 0.002 | 0.048 ± 0.001 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.086 ± 0.001 | 0.051 ± 0.003 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.072 ± 0.004 | 0.051 ± 0.001 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.104 ± 0.001 | 0.063 ± 0.001 | 0.042 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.100 ± 0.000 | 0.058 ± 0.002 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.096 ± 0.017 | 0.063 ± 0.002 | 0.036 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.090 ± 0.019 | 0.058 ± 0.002 | 0.017 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.108 ± 0.007 | 0.068 ± 0.000 | 0.052 ± 0.001 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.205 ± 0.003 | 0.209 ± 0.003 | 0.185 ± 0.007 | 0.207 ± 0.003 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 16.881 ± 9.516 | 4.834 ± 13.758 | 6.303 ± 0.315 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 19.859 ± 0.421 | 26.791 ± 0.571 | 29.414 ± 4.032 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 15.573 ± 0.237 | 3.642 ± 0.077 | 6.463 ± 0.124 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 20.919 ± 0.433 | 19.828 ± 0.742 | 33.821 ± 4.649 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 3.340 ± 0.086 | 5.139 ± 0.099 | 3.642 ± 0.427 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 12.656 ± 0.168 | 32.686 ± 0.545 | 21.306 ± 0.952 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 3.296 ± 0.109 | 2.493 ± 0.082 | 3.286 ± 0.602 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 12.903 ± 1.232 | 16.559 ± 0.259 | 21.248 ± 2.835 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.046 ± 0.002 | 0.050 ± 0.001 | 0.021 ± 0.001 | 0.021 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.150 ± 0.006 | 0.074 ± 0.012 | 0.056 ± 0.009 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 5.470 ± 0.060 | 3.154 ± 0.051 | 5.241 ± 1.321 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 20.473 ± 0.245 | 23.068 ± 0.468 | 31.940 ± 1.720 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 5.265 ± 0.144 | 3.166 ± 0.132 | 5.148 ± 0.281 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 19.627 ± 0.136 | 22.348 ± 0.407 | 32.800 ± 2.017 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.143 ± 0.003 | 0.059 ± 0.002 | 0.044 ± 0.004 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.620 ± 0.010 | 2.982 ± 1.671 | 0.471 ± 0.227 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.445 ± 0.038 | 3.070 ± 4.038 | 1.942 ± 0.086 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 0.586 ± 0.021 | 0.769 ± 0.018 | 0.497 ± 0.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 1.379 ± 0.038 | 2.886 ± 0.056 | 2.064 ± 0.116 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.496 ± 0.011 | 0.511 ± 0.014 | 0.456 ± 0.012 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 19.022 ± 0.224 | 12.224 ± 0.453 | 9.712 ± 0.580 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 46.865 ± 1.301 | 59.170 ± 0.557 | 66.659 ± 4.127 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 14.035 ± 0.297 | 9.480 ± 0.392 | 10.409 ± 1.108 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 44.943 ± 0.879 | 52.007 ± 0.891 | 63.097 ± 3.030 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.057 ± 0.003 | 0.171 ± 0.004 | 0.090 ± 0.006 | 0.109 ± 0.016 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.324 ± 0.010 | 0.396 ± 0.063 | 0.425 ± 0.040 | 0.460 ± 0.138 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.250 ± 0.009 | 1.430 ± 0.011 | 1.650 ± 0.329 | 1.844 ± 0.177 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.104 ± 0.001 | 0.105 ± 0.003 | 0.092 ± 0.005 | 0.070 ± 0.006 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.030 ± 0.001 | 0.049 ± 0.001 | 0.028 ± 0.001 | 0.036 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.035 ± 0.001 | 0.065 ± 0.002 | 0.049 ± 0.005 | 0.073 ± 0.013 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.051 ± 0.005 | 0.077 ± 0.006 | 0.064 ± 0.012 | 0.096 ± 0.005 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.449 ± 0.007 | 0.436 ± 0.009 | 0.433 ± 0.013 | 0.414 ± 0.010 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.014 ± 0.001 | 0.015 ± 0.001 | 0.004 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.015 ± 0.001 | 0.017 ± 0.001 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.019 ± 0.002 | 0.020 ± 0.001 | 0.008 ± 0.001 | 0.011 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.065 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.071 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.090 ± 0.019 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.139 ± 0.004 | 0.057 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.144 ± 0.022 | 0.059 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.128 ± 0.022 | 0.064 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.110 ± 0.008 | 0.048 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.131 ± 0.011 | 0.050 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.140 ± 0.004 | 0.052 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.101 ± 0.012 | 0.144 ± 0.003 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.107 ± 0.008 | 0.143 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.117 ± 0.001 | 0.146 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.093 ± 0.018 | 0.107 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.111 ± 0.002 | 0.113 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.110 ± 0.004 | 0.112 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.065 ± 0.000 | 0.019 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.074 ± 0.001 | 0.021 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.001 | 0.021 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.232 ± 0.038 | 0.098 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.222 ± 0.002 | 0.103 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.227 ± 0.003 | 0.102 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.186 ± 0.008 | 0.099 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.186 ± 0.047 | 0.101 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.218 ± 0.020 | 0.104 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.089 ± 0.004 | 0.030 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.077 ± 0.001 | 0.031 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.091 ± 0.001 | 0.032 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.074 ± 0.003 | 0.153 ± 0.005 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.072 ± 0.013 | 0.149 ± 0.002 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.073 ± 0.011 | 0.151 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.068 ± 0.004 | 0.064 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.058 ± 0.010 | 0.065 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.068 ± 0.001 | 0.067 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.001 | 0.022 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.057 ± 0.001 | 0.026 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.065 ± 0.006 | 0.031 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.052 ± 0.001 | 0.086 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.056 ± 0.001 | 0.089 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.075 ± 0.002 | 0.101 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.051 ± 0.004 | 0.053 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.049 ± 0.007 | 0.056 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.066 ± 0.002 | 0.063 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.026 ± 0.001 | 0.001 ± 0.000 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.008 ± 0.001 | 0.035 ± 0.002 | 0.007 ± 0.000 | 0.009 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.002 ± 0.000 | 0.021 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.015 ± 0.000 | 0.043 ± 0.005 | 0.022 ± 0.002 | 0.023 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.023 ± 0.001 | 0.002 ± 0.000 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.013 ± 0.001 | 0.014 ± 0.001 | 0.005 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.006 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.014 ± 0.001 | 0.015 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.022 ± 0.002 | 0.008 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.003 ± 0.000 | 0.021 ± 0.002 | 0.008 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.023 ± 0.001 | 0.008 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.021 ± 0.000 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.023 ± 0.001 | 0.009 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.024 ± 0.001 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.019 ± 0.001 | 0.020 ± 0.002 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.027 ± 0.001 | 0.029 ± 0.002 | 0.013 ± 0.001 | 0.016 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.077 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 32.1x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.3x faster than the slowest successful cell (`tenferro-trace`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 0.128 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`tenferro-trace`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.7x faster than the slowest successful cell (`tenferro-trace`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 37.7x faster than the slowest successful cell (`tenferro-trace`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 32.2x faster than the slowest successful cell (`tenferro-trace`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-trace`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 31.8x faster than the slowest successful cell (`tenferro-trace`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 27.5x faster than the slowest successful cell (`tenferro-trace`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-trace`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.089 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 0.026 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 0.021 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`tenferro-trace`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

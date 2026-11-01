# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`
- Benchmark commit: `ab8e941ebccb8606e3781e9d8f6bddeeeab3d2a2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261101_180030/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_180030/run.yaml`
- Timestamp: `2026-11-01T18:00:30.835405+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261101_180030`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_180030/cpu_ops_t1_20261101_180030.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.028 ± 0.001 | 0.030 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.032 ± 0.000 | 0.029 ± 0.000 | 0.032 ± 0.008 | 0.029 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.102 ± 0.002 | 0.103 ± 0.001 | 0.094 ± 0.001 | 0.099 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.002 | 0.103 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.112 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.100 ± 0.003 | 0.092 ± 0.004 | 0.133 ± 0.015 | 0.135 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.110 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.112 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.029 ± 0.001 | 0.028 ± 0.000 | 0.040 ± 0.000 | 0.049 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.027 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.018 ± 0.000 | 0.019 ± 0.000 | 0.024 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.059 ± 0.002 | 0.057 ± 0.000 | 0.056 ± 0.000 | 0.062 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.065 ± 0.002 | 0.057 ± 0.001 | 0.053 ± 0.000 | 0.054 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.217 ± 0.003 | 0.211 ± 0.005 | 0.191 ± 0.002 | 0.203 ± 0.047 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.012 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.022 ± 0.000 | 0.063 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.065 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.091 ± 0.005 | 0.035 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.020 ± 0.000 | 0.047 ± 0.000 | 0.018 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.055 ± 0.001 | 0.040 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.049 ± 0.001 | 0.029 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.062 ± 0.001 | 0.054 ± 0.005 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.219 ± 0.005 | 0.186 ± 0.006 | 0.194 ± 0.007 | 0.262 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | - | 4.542 ± 0.059 | 5.063 ± 0.041 | 6.167 ± 0.065 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | - | 25.200 ± 0.263 | 30.473 ± 3.628 | 42.410 ± 4.305 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | - | 4.527 ± 0.050 | 4.277 ± 0.061 | 6.105 ± 0.050 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | - | 25.199 ± 0.491 | 27.598 ± 2.425 | 38.443 ± 5.209 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | - | 2.115 ± 0.011 | 4.436 ± 0.048 | 4.294 ± 0.046 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | - | 14.606 ± 0.066 | 35.715 ± 0.328 | 27.335 ± 0.344 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | - | 2.218 ± 0.014 | 2.603 ± 0.005 | 4.291 ± 0.102 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | - | 14.919 ± 0.129 | 19.091 ± 0.438 | 29.325 ± 0.324 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.024 ± 0.001 | 0.011 ± 0.000 | 0.021 ± 0.001 | 0.040 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.001 | 0.078 ± 0.007 | 0.085 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | - | 3.172 ± 0.051 | 3.581 ± 0.024 | 5.687 ± 0.085 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | - | 21.540 ± 0.406 | 23.563 ± 0.311 | 34.053 ± 4.117 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | - | 3.417 ± 0.063 | 3.507 ± 0.135 | 5.525 ± 0.155 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | - | 22.021 ± 0.168 | 22.868 ± 0.715 | 45.540 ± 4.597 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.039 ± 0.001 | 0.088 ± 0.002 | 0.059 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.359 ± 0.013 | 0.764 ± 0.026 | 0.565 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.101 ± 0.018 | 3.069 ± 0.088 | 3.674 ± 0.022 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | - | 0.378 ± 0.022 | 0.974 ± 0.017 | 0.581 ± 0.034 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | - | 2.112 ± 0.023 | 4.831 ± 0.109 | 3.328 ± 0.092 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.392 ± 0.007 | 0.457 ± 0.010 | 0.445 ± 0.001 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | - | 7.591 ± 0.130 | 9.321 ± 0.724 | 10.347 ± 0.139 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | - | 49.477 ± 0.755 | 70.566 ± 9.547 | 70.895 ± 0.305 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | - | 7.657 ± 0.036 | 7.293 ± 0.394 | 10.262 ± 0.016 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | - | 50.762 ± 0.561 | 49.431 ± 2.213 | 75.247 ± 3.473 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.064 ± 0.001 | 0.055 ± 0.001 | 0.115 ± 0.010 | 0.173 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.444 ± 0.004 | 0.396 ± 0.003 | 0.596 ± 0.024 | 1.159 ± 0.029 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 1.794 ± 0.013 | 1.605 ± 0.038 | 1.894 ± 0.072 | 3.965 ± 0.146 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.054 ± 0.001 | 0.047 ± 0.001 | 0.063 ± 0.001 | 0.085 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.025 ± 0.000 | 0.023 ± 0.000 | 0.036 ± 0.001 | 0.042 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.025 ± 0.001 | 0.026 ± 0.003 | 0.042 ± 0.003 | 0.052 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.034 ± 0.002 | 0.033 ± 0.000 | 0.056 ± 0.004 | 0.163 ± 0.015 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.426 ± 0.005 | 0.369 ± 0.005 | 0.377 ± 0.015 | 0.410 ± 0.006 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.013 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.098 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.090 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.113 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | - | 0.015 ± 0.001 | 0.082 ± 0.003 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | - | 0.017 ± 0.000 | 0.087 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | - | 0.020 ± 0.000 | 0.107 ± 0.005 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | - | 0.015 ± 0.000 | 0.078 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | - | 0.018 ± 0.000 | 0.082 ± 0.015 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | - | 0.019 ± 0.001 | 0.095 ± 0.004 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | - | 0.018 ± 0.000 | 0.129 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | - | 0.022 ± 0.002 | 0.129 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | - | 0.019 ± 0.000 | 0.145 ± 0.019 | 0.024 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | - | 0.017 ± 0.000 | 0.108 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | - | 0.019 ± 0.000 | 0.111 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | - | 0.018 ± 0.000 | 0.112 ± 0.011 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.027 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.029 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.008 ± 0.000 | 0.029 ± 0.000 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | - | 0.026 ± 0.000 | 0.096 ± 0.002 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | - | 0.026 ± 0.001 | 0.098 ± 0.007 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | - | 0.027 ± 0.000 | 0.096 ± 0.004 | 0.014 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | - | 0.025 ± 0.000 | 0.103 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | - | 0.025 ± 0.000 | 0.106 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | - | 0.027 ± 0.000 | 0.106 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.042 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.045 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.012 ± 0.000 | 0.216 ± 0.031 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.013 ± 0.000 | 0.213 ± 0.008 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.012 ± 0.000 | 0.217 ± 0.013 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | - | 0.012 ± 0.000 | 0.104 ± 0.011 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | - | 0.013 ± 0.000 | 0.100 ± 0.001 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | - | 0.012 ± 0.000 | 0.100 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.011 ± 0.001 | 0.034 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.038 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.046 ± 0.001 | 0.024 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | - | 0.011 ± 0.000 | 0.121 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | - | 0.014 ± 0.000 | 0.125 ± 0.003 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | - | 0.022 ± 0.000 | 0.138 ± 0.019 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | - | 0.011 ± 0.000 | 0.085 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | - | 0.016 ± 0.001 | 0.090 ± 0.003 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | - | 0.021 ± 0.001 | 0.096 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.011 ± 0.002 | 0.010 ± 0.000 | 0.025 ± 0.002 | 0.038 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.010 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.017 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.020 ± 0.004 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 29.9x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 25.1x faster than the slowest successful cell (`jax-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 40.1x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 34.8x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 41.4x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 26.1x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.213 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261101_183239/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261101_183239/run.yaml`
- Timestamp: `2026-11-01T18:32:39.800639+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261101_183239`.

- tenferro-rs commit: `a9f494cc4679387c75af698b299c316fd2295fe4`

### CPU Information

- Model: `Intel(R) Xeon(R) 6973P-C`
- Vendor: `GenuineIntel`
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


- CSV: `data/results/amd-cpu/cpu/einsum/20261101_183239/cpu_ops_t4_20261101_183239.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.030 ± 0.005 | 0.032 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.030 ± 0.000 | 0.029 ± 0.000 | 0.028 ± 0.000 | 0.030 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.104 ± 0.001 | 0.101 ± 0.001 | 0.105 ± 0.003 | 0.101 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.094 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.133 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.148 ± 0.003 | 0.149 ± 0.003 | 0.089 ± 0.021 | 0.206 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.109 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.125 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.001 | 0.011 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.022 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.029 ± 0.000 | 0.029 ± 0.000 | 0.053 ± 0.001 | 0.059 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.000 | 0.023 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.000 | 0.034 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.057 ± 0.001 | 0.057 ± 0.001 | 0.054 ± 0.001 | 0.070 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.059 ± 0.001 | 0.058 ± 0.000 | 0.053 ± 0.006 | 0.053 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.217 ± 0.001 | 0.216 ± 0.002 | 0.190 ± 0.008 | 0.186 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.060 ± 0.000 | 0.011 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.001 | 0.064 ± 0.002 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.001 | 0.064 ± 0.001 | 0.020 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.086 ± 0.003 | 0.050 ± 0.007 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.025 ± 0.001 | 0.060 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.034 ± 0.000 | 0.068 ± 0.001 | 0.043 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.027 ± 0.000 | 0.063 ± 0.000 | 0.027 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.040 ± 0.000 | 0.080 ± 0.002 | 0.064 ± 0.007 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.205 ± 0.002 | 0.222 ± 0.003 | 0.200 ± 0.319 | 0.283 ± 0.005 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | - | 49.995 ± 2.004 | 4.288 ± 0.361 | 7.533 ± 0.251 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | - | 57.160 ± 2.996 | 20.458 ± 0.861 | 39.962 ± 7.008 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | - | 49.998 ± 3.991 | 3.530 ± 0.073 | 7.021 ± 0.461 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | - | 24.213 ± 2.870 | 15.072 ± 2.172 | 35.963 ± 4.421 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | - | 53.001 ± 3.992 | 2.918 ± 0.048 | 4.476 ± 0.408 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | - | 27.372 ± 22.330 | 21.399 ± 3.970 | 26.340 ± 6.367 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | - | 42.058 ± 1.243 | 1.886 ± 0.096 | 4.130 ± 1.180 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | - | 39.860 ± 23.810 | 11.723 ± 0.157 | 28.557 ± 3.930 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.033 ± 0.003 | 0.015 ± 0.002 | 0.022 ± 0.000 | 0.042 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.010 | 0.084 ± 0.008 | 0.084 ± 0.005 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | - | 89.996 ± 7.000 | 3.540 ± 0.742 | 7.126 ± 0.145 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | - | 28.097 ± 62.769 | 14.026 ± 1.235 | 39.381 ± 6.982 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | - | 87.005 ± 3.002 | 3.028 ± 0.009 | 7.360 ± 0.990 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | - | 94.988 ± 2.761 | 13.482 ± 0.922 | 40.156 ± 7.186 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.062 ± 0.001 | 0.088 ± 0.002 | 0.066 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | - | 13.997 ± 0.006 | 2.591 ± 1.937 | 0.725 ± 0.171 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | - | 9.993 ± 6.941 | 2.055 ± 3.859 | 3.345 ± 0.515 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | - | 15.942 ± 1.353 | 0.813 ± 0.047 | 0.759 ± 0.064 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | - | 14.902 ± 1.726 | 2.916 ± 0.084 | 3.135 ± 0.067 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.527 ± 0.003 | 0.578 ± 0.045 | 0.629 ± 0.024 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | - | 16.078 ± 0.634 | 8.511 ± 0.308 | 12.981 ± 0.725 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | - | 47.985 ± 2.862 | 47.232 ± 5.623 | 77.968 ± 9.952 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | - | 22.999 ± 1.292 | 8.557 ± 0.940 | 12.389 ± 0.480 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | - | 58.595 ± 3.877 | 40.208 ± 7.051 | 73.949 ± 6.772 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.038 ± 0.002 | 0.034 ± 0.002 | 0.073 ± 0.004 | 0.145 ± 0.012 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.220 ± 0.006 | 0.223 ± 0.013 | 0.314 ± 0.022 | 0.605 ± 0.074 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.851 ± 0.015 | 0.861 ± 0.017 | 1.030 ± 0.035 | 2.464 ± 0.103 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.075 ± 0.001 | 0.076 ± 0.003 | 0.076 ± 0.001 | 0.100 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.029 ± 0.001 | 0.033 ± 0.003 | 0.035 ± 0.001 | 0.045 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.037 ± 0.002 | 0.050 ± 0.002 | 0.060 ± 0.004 | 0.113 ± 0.093 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.052 ± 0.002 | 0.059 ± 0.007 | 0.076 ± 0.005 | 0.147 ± 0.007 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.447 ± 0.010 | 0.430 ± 0.017 | 0.432 ± 0.030 | 0.564 ± 0.012 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.092 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.093 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.117 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | - | 0.021 ± 0.001 | 0.087 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | - | 0.022 ± 0.001 | 0.089 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | - | 0.026 ± 0.001 | 0.094 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | - | 0.020 ± 0.000 | 0.079 ± 0.001 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | - | 0.021 ± 0.000 | 0.082 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.084 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | - | 0.025 ± 0.000 | 0.185 ± 0.023 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | - | 0.025 ± 0.001 | 0.170 ± 0.004 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | - | 0.026 ± 0.000 | 0.172 ± 0.004 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | - | 0.023 ± 0.000 | 0.157 ± 0.026 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | - | 0.023 ± 0.001 | 0.139 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | - | 0.024 ± 0.001 | 0.138 ± 0.001 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.032 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.030 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.029 ± 0.001 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | - | 0.035 ± 0.000 | 0.122 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | - | 0.036 ± 0.001 | 0.123 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | - | 0.037 ± 0.000 | 0.139 ± 0.002 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | - | 0.034 ± 0.000 | 0.125 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | - | 0.035 ± 0.000 | 0.127 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | - | 0.037 ± 0.001 | 0.128 ± 0.017 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.001 | 0.048 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.044 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.001 | 0.050 ± 0.017 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.244 ± 0.050 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.017 ± 0.000 | 0.215 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.017 ± 0.000 | 0.216 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | - | 0.016 ± 0.000 | 0.150 ± 0.076 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | - | 0.016 ± 0.000 | 0.101 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | - | 0.017 ± 0.000 | 0.103 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.046 ± 0.019 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.016 ± 0.000 | 0.039 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.053 ± 0.009 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | - | 0.014 ± 0.000 | 0.137 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | - | 0.017 ± 0.000 | 0.129 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | - | 0.025 ± 0.000 | 0.157 ± 0.007 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | - | 0.013 ± 0.000 | 0.088 ± 0.002 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | - | 0.016 ± 0.000 | 0.089 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | - | 0.023 ± 0.001 | 0.107 ± 0.013 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.012 ± 0.000 | 0.012 ± 0.000 | 0.025 ± 0.001 | 0.047 ± 0.007 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.007 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.133 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 23.6x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_eigh_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 49.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_eigh_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`tenferro-trace`, 49.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 53.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 53.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 42.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`tenferro-trace`, 42.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 12.6x faster than the slowest successful cell (`tenferro-trace`, 89.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 25.4x faster than the slowest successful cell (`tenferro-trace`, 89.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 87.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 28.7x faster than the slowest successful cell (`tenferro-trace`, 87.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`tenferro-trace`, 13.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-trace`, 15.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`256x256,rhs=1`): `pytorch-cpu` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 15.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 30.3x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.093 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 37.3x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.123 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 30.1x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 26.1x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.7x faster than the slowest successful cell (`pytorch-cpu`, 0.137 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

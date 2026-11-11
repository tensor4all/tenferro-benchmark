# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `35979dcd7110a29d0728b2e128c72914f4c3f67b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261111_180426/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261111_180426/run.yaml`
- Timestamp: `2026-11-11T18:04:26.325001+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261111_180426`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261111_180426/cpu_ops_t1_20261111_180426.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.035 ± 0.000 | 0.034 ± 0.000 | 0.029 ± 0.000 | 0.033 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.032 ± 0.000 | 0.031 ± 0.000 | 0.031 ± 0.000 | 0.032 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.113 ± 0.000 | 0.112 ± 0.000 | 0.103 ± 0.001 | 0.106 ± 0.000 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.107 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.117 ± 0.026 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.104 ± 0.001 | 0.103 ± 0.000 | 0.127 ± 0.005 | 0.132 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.118 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.146 ± 0.021 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | 0.019 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.031 ± 0.001 | 0.031 ± 0.000 | 0.032 ± 0.001 | 0.054 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.021 ± 0.000 | 0.023 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.070 ± 0.001 | 0.068 ± 0.000 | 0.061 ± 0.000 | 0.070 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.063 ± 0.000 | 0.062 ± 0.000 | 0.058 ± 0.000 | 0.057 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.234 ± 0.001 | 0.232 ± 0.001 | 0.205 ± 0.002 | 0.194 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.014 ± 0.000 | 0.087 ± 0.001 | 0.010 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.026 ± 0.000 | 0.091 ± 0.001 | 0.012 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.092 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.112 ± 0.002 | 0.037 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.037 ± 0.000 | 0.077 ± 0.002 | 0.018 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.060 ± 0.000 | 0.088 ± 0.003 | 0.042 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.038 ± 0.000 | 0.083 ± 0.003 | 0.020 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.063 ± 0.000 | 0.095 ± 0.003 | 0.043 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.244 ± 0.000 | 0.243 ± 0.003 | 0.253 ± 0.003 | 0.280 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 15.961 ± 0.152 | 9.917 ± 0.181 | 8.766 ± 0.108 | 7.045 ± 0.023 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 95.201 ± 1.138 | 64.337 ± 0.390 | 57.130 ± 0.596 | 44.701 ± 0.188 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.001 ± 0.120 | 9.871 ± 0.049 | 7.096 ± 0.189 | 7.006 ± 0.017 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 56.526 ± 0.198 | 64.527 ± 0.619 | 44.939 ± 0.495 | 44.293 ± 0.079 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 9.208 ± 0.081 | 4.656 ± 0.025 | 10.695 ± 0.632 | 4.103 ± 0.009 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 64.642 ± 1.811 | 32.866 ± 0.598 | 70.092 ± 0.806 | 26.951 ± 0.061 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 8.577 ± 0.085 | 4.912 ± 0.035 | 4.518 ± 0.024 | 3.973 ± 0.022 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 60.784 ± 0.721 | 33.959 ± 0.588 | 36.310 ± 0.215 | 26.747 ± 3.776 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.001 | 0.020 ± 0.000 | 0.030 ± 0.001 | 0.029 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.060 ± 0.001 | 0.103 ± 0.001 | 0.067 ± 0.000 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 14.828 ± 0.037 | 8.844 ± 0.042 | 6.249 ± 0.013 | 5.591 ± 0.032 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 102.339 ± 0.435 | 63.927 ± 0.288 | 45.533 ± 0.835 | 36.784 ± 2.729 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 10.803 ± 0.167 | 8.568 ± 0.030 | 6.189 ± 0.175 | 5.543 ± 0.024 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 78.663 ± 1.298 | 60.710 ± 1.256 | 45.846 ± 1.618 | 40.692 ± 0.054 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.105 ± 0.004 | 0.114 ± 0.002 | 0.048 ± 0.000 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.581 ± 0.015 | 1.090 ± 0.002 | 1.015 ± 0.015 | 0.574 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 7.409 ± 0.138 | 6.773 ± 0.013 | 4.631 ± 0.107 | 3.501 ± 0.034 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.694 ± 0.003 | 1.097 ± 0.003 | 1.404 ± 0.021 | 0.574 ± 0.009 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 3.870 ± 0.149 | 6.727 ± 0.038 | 7.973 ± 0.310 | 3.596 ± 0.016 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.667 ± 0.003 | 0.620 ± 0.005 | 0.468 ± 0.006 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 24.681 ± 0.277 | 16.696 ± 0.084 | 18.078 ± 0.483 | 12.258 ± 0.778 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 166.921 ± 0.667 | 118.871 ± 0.307 | 119.795 ± 0.959 | 87.578 ± 0.898 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 12.887 ± 0.253 | 16.676 ± 0.048 | 13.455 ± 0.077 | 12.199 ± 0.038 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 88.075 ± 0.690 | 118.578 ± 0.279 | 95.427 ± 0.569 | 92.590 ± 0.162 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.113 ± 0.002 | 0.110 ± 0.003 | 0.181 ± 0.007 | 0.145 ± 0.002 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.784 ± 0.011 | 0.700 ± 0.012 | 0.953 ± 0.007 | 0.908 ± 0.005 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 2.819 ± 0.020 | 2.839 ± 0.035 | 3.024 ± 0.031 | 3.306 ± 0.063 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.074 ± 0.000 | 0.072 ± 0.000 | 0.092 ± 0.002 | 0.083 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.027 ± 0.000 | 0.029 ± 0.000 | 0.039 ± 0.000 | 0.035 ± 0.004 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.036 ± 0.000 | 0.038 ± 0.001 | 0.053 ± 0.002 | 0.053 ± 0.017 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.063 ± 0.000 | 0.067 ± 0.002 | 0.095 ± 0.003 | 0.118 ± 0.005 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.451 ± 0.002 | 0.448 ± 0.003 | 0.494 ± 0.007 | 0.449 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.102 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.104 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.117 ± 0.027 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.191 ± 0.005 | 0.021 ± 0.000 | 0.118 ± 0.004 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.200 ± 0.001 | 0.024 ± 0.000 | 0.122 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.209 ± 0.002 | 0.029 ± 0.000 | 0.128 ± 0.001 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.061 ± 0.004 | 0.021 ± 0.000 | 0.114 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.067 ± 0.002 | 0.024 ± 0.000 | 0.121 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.072 ± 0.001 | 0.029 ± 0.000 | 0.125 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.553 ± 0.004 | 0.026 ± 0.000 | 0.182 ± 0.018 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.561 ± 0.003 | 0.026 ± 0.000 | 0.178 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.569 ± 0.012 | 0.027 ± 0.000 | 0.180 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.137 ± 0.003 | 0.024 ± 0.000 | 0.166 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.139 ± 0.002 | 0.024 ± 0.000 | 0.168 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.142 ± 0.001 | 0.026 ± 0.000 | 0.167 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.035 ± 0.001 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.036 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.010 ± 0.000 | 0.036 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.908 ± 0.150 | 0.037 ± 0.000 | 0.137 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.783 ± 0.003 | 0.038 ± 0.000 | 0.136 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.788 ± 0.005 | 0.040 ± 0.000 | 0.139 ± 0.003 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.133 ± 0.001 | 0.037 ± 0.000 | 0.161 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.135 ± 0.000 | 0.039 ± 0.001 | 0.161 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.140 ± 0.007 | 0.041 ± 0.000 | 0.162 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.028 ± 0.000 | 0.063 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.071 ± 0.021 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.080 ± 0.029 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.138 ± 0.001 | 0.020 ± 0.000 | 0.274 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.139 ± 0.001 | 0.020 ± 0.000 | 0.275 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.140 ± 0.000 | 0.021 ± 0.000 | 0.275 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.058 ± 0.000 | 0.020 ± 0.000 | 0.154 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.058 ± 0.001 | 0.020 ± 0.000 | 0.154 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.060 ± 0.001 | 0.021 ± 0.000 | 0.155 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.015 ± 0.000 | 0.061 ± 0.001 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.019 ± 0.000 | 0.068 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.030 ± 0.000 | 0.077 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.164 ± 0.002 | 0.016 ± 0.000 | 0.179 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.179 ± 0.002 | 0.020 ± 0.000 | 0.184 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.197 ± 0.001 | 0.031 ± 0.000 | 0.196 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.054 ± 0.000 | 0.015 ± 0.000 | 0.131 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.061 ± 0.000 | 0.019 ± 0.000 | 0.138 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.071 ± 0.000 | 0.030 ± 0.000 | 0.148 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.001 | 0.019 ± 0.000 | 0.039 ± 0.000 | 0.038 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.002 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.007 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 24.4x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 26.0x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.7x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 32.8x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.102 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 33.4x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 31.9x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 21.8x faster than the slowest successful cell (`tenferro-eager`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.4x faster than the slowest successful cell (`tenferro-eager`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.114 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 63.3x faster than the slowest successful cell (`tenferro-eager`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 0.553 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 62.4x faster than the slowest successful cell (`tenferro-eager`, 0.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`tenferro-eager`, 0.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 42.3x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.569 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 104.6x faster than the slowest successful cell (`tenferro-eager`, 0.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`tenferro-eager`, 0.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 84.3x faster than the slowest successful cell (`tenferro-eager`, 0.783 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.783 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 53.4x faster than the slowest successful cell (`tenferro-eager`, 0.788 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.788 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 31.9x faster than the slowest successful cell (`pytorch-cpu`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 31.3x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 29.3x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.275 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.8x faster than the slowest successful cell (`pytorch-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.197 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261111_183853/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261111_183853/run.yaml`
- Timestamp: `2026-11-11T18:38:53.952407+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261111_183853`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261111_183853/cpu_ops_t4_20261111_183853.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal rows reuse a borrowed session entered before sampling; eager backward rows without a borrowed-session API are unsupported. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.034 ± 0.000 | 0.033 ± 0.000 | 0.028 ± 0.000 | 0.033 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.032 ± 0.000 | 0.032 ± 0.000 | 0.030 ± 0.000 | 0.032 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.113 ± 0.001 | 0.112 ± 0.000 | 0.101 ± 0.002 | 0.108 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.104 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.118 ± 0.034 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.105 ± 0.000 | 0.105 ± 0.000 | 0.084 ± 0.001 | 0.195 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.154 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.115 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.012 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.024 ± 0.001 | 0.022 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.002 | 0.018 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.033 ± 0.000 | 0.032 ± 0.000 | 0.046 ± 0.001 | 0.048 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.000 | 0.011 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.029 ± 0.001 | 0.025 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.023 ± 0.000 | 0.013 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.019 ± 0.000 | 0.029 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.024 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.068 ± 0.001 | 0.068 ± 0.000 | 0.058 ± 0.000 | 0.071 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.063 ± 0.000 | 0.063 ± 0.000 | 0.058 ± 0.001 | 0.057 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.232 ± 0.001 | 0.236 ± 0.001 | 0.204 ± 0.002 | 0.195 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.017 ± 0.000 | 0.088 ± 0.008 | 0.010 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.029 ± 0.000 | 0.091 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.018 ± 0.000 | 0.092 ± 0.001 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.033 ± 0.000 | 0.110 ± 0.003 | 0.058 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.050 ± 0.000 | 0.097 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.076 ± 0.000 | 0.110 ± 0.000 | 0.039 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.053 ± 0.000 | 0.101 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.085 ± 0.001 | 0.124 ± 0.001 | 0.050 ± 0.019 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.418 ± 0.006 | 0.414 ± 0.002 | 0.264 ± 0.008 | 0.287 ± 0.001 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 29.092 ± 0.232 | 21.839 ± 0.073 | 5.881 ± 0.148 | 8.940 ± 1.292 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 130.034 ± 0.252 | 98.602 ± 0.326 | 35.538 ± 1.022 | 44.696 ± 1.687 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 15.667 ± 0.181 | 21.711 ± 0.063 | 5.011 ± 0.176 | 9.009 ± 0.109 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 72.231 ± 0.549 | 98.403 ± 0.192 | 25.200 ± 0.985 | 44.045 ± 2.955 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 14.653 ± 0.209 | 8.086 ± 0.023 | 8.765 ± 2.776 | 6.988 ± 0.002 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 280.501 ± 12.119 | 226.922 ± 98.907 | 40.112 ± 0.520 | 30.598 ± 2.023 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 12.524 ± 0.094 | 8.268 ± 0.087 | 3.008 ± 0.021 | 4.813 ± 0.996 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 190.752 ± 23.917 | 224.808 ± 77.524 | 20.428 ± 0.203 | 24.968 ± 3.314 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.056 ± 0.001 | 0.044 ± 0.000 | 0.031 ± 0.001 | 0.032 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.129 ± 0.000 | 0.122 ± 0.001 | 0.084 ± 0.007 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 113.715 ± 32.760 | 12.892 ± 0.026 | 4.074 ± 0.086 | 7.487 ± 0.830 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 419.060 ± 14.219 | 383.885 ± 10.265 | 28.460 ± 0.331 | 43.582 ± 6.176 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 15.363 ± 6.796 | 12.658 ± 0.040 | 4.680 ± 0.091 | 8.707 ± 0.396 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 243.621 ± 17.736 | 367.621 ± 12.780 | 28.239 ± 1.982 | 38.428 ± 12.119 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.117 ± 0.001 | 0.112 ± 0.001 | 0.070 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 3.306 ± 0.016 | 2.711 ± 0.046 | 1.528 ± 0.083 | 0.711 ± 0.042 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 178.809 ± 13.044 | 165.957 ± 95.011 | 2.483 ± 0.189 | 2.953 ± 0.094 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.541 ± 0.026 | 2.684 ± 0.010 | 1.002 ± 0.052 | 0.805 ± 0.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 102.949 ± 11.908 | 258.008 ± 5.958 | 3.991 ± 0.020 | 3.177 ± 0.029 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 1.219 ± 0.007 | 0.822 ± 0.003 | 0.671 ± 0.069 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 31.546 ± 0.318 | 20.605 ± 0.026 | 13.943 ± 0.075 | 13.959 ± 3.409 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 202.799 ± 1.218 | 150.622 ± 0.208 | 79.447 ± 1.447 | 79.962 ± 2.123 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 17.365 ± 0.140 | 22.040 ± 0.111 | 12.763 ± 0.052 | 14.289 ± 1.278 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 107.582 ± 0.208 | 151.981 ± 0.590 | 70.082 ± 1.383 | 84.961 ± 6.152 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.168 ± 0.067 | 0.146 ± 0.001 | 0.125 ± 0.001 | 0.135 ± 0.037 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.841 ± 0.006 | 0.790 ± 0.003 | 0.503 ± 0.011 | 0.534 ± 0.110 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 3.085 ± 0.025 | 3.035 ± 0.005 | 1.642 ± 0.019 | 2.306 ± 0.128 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.226 ± 0.002 | 0.225 ± 0.000 | 0.144 ± 0.001 | 0.085 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.028 ± 0.000 | 0.030 ± 0.000 | 0.037 ± 0.001 | 0.043 ± 0.001 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.060 ± 0.002 | 0.063 ± 0.001 | 0.072 ± 0.003 | 0.110 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.088 ± 0.002 | 0.094 ± 0.004 | 0.107 ± 0.002 | 0.154 ± 0.006 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.887 ± 0.003 | 0.876 ± 0.008 | 0.690 ± 0.007 | 0.625 ± 0.033 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.099 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.106 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.146 ± 0.036 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.201 ± 0.002 | 0.030 ± 0.000 | 0.119 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.212 ± 0.001 | 0.033 ± 0.000 | 0.123 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.222 ± 0.001 | 0.038 ± 0.000 | 0.131 ± 0.003 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.072 ± 0.001 | 0.030 ± 0.000 | 0.112 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.080 ± 0.001 | 0.033 ± 0.000 | 0.119 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.086 ± 0.001 | 0.038 ± 0.000 | 0.124 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.584 ± 0.004 | 0.037 ± 0.001 | 0.248 ± 0.004 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.586 ± 0.004 | 0.038 ± 0.002 | 0.246 ± 0.004 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.604 ± 0.000 | 0.039 ± 0.000 | 0.250 ± 0.011 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.174 ± 0.002 | 0.034 ± 0.000 | 0.207 ± 0.007 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.175 ± 0.001 | 0.034 ± 0.000 | 0.208 ± 0.004 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.179 ± 0.003 | 0.035 ± 0.000 | 0.210 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.035 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.038 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.013 ± 0.000 | 0.039 ± 0.001 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.774 ± 0.012 | 0.052 ± 0.000 | 0.178 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.782 ± 0.004 | 0.054 ± 0.000 | 0.178 ± 0.004 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.824 ± 0.005 | 0.063 ± 0.012 | 0.181 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.166 ± 0.001 | 0.052 ± 0.000 | 0.198 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.167 ± 0.001 | 0.054 ± 0.000 | 0.198 ± 0.005 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.172 ± 0.007 | 0.056 ± 0.000 | 0.200 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.064 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.041 ± 0.000 | 0.066 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.043 ± 0.000 | 0.067 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.148 ± 0.001 | 0.028 ± 0.000 | 0.273 ± 0.005 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.149 ± 0.002 | 0.028 ± 0.000 | 0.271 ± 0.003 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.153 ± 0.004 | 0.029 ± 0.000 | 0.274 ± 0.002 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.068 ± 0.000 | 0.027 ± 0.000 | 0.151 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.068 ± 0.002 | 0.028 ± 0.000 | 0.151 ± 0.001 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.071 ± 0.014 | 0.029 ± 0.000 | 0.153 ± 0.009 | 0.010 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.021 ± 0.000 | 0.060 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.024 ± 0.000 | 0.067 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | unsupported: eager backward has no borrowed-session API; setup-inclusive timings disabled | 0.035 ± 0.000 | 0.077 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.173 ± 0.003 | 0.023 ± 0.000 | 0.178 ± 0.001 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.185 ± 0.006 | 0.025 ± 0.000 | 0.185 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.207 ± 0.007 | 0.036 ± 0.001 | 0.197 ± 0.004 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.063 ± 0.001 | 0.020 ± 0.001 | 0.132 ± 0.002 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.069 ± 0.001 | 0.024 ± 0.000 | 0.138 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.081 ± 0.002 | 0.035 ± 0.000 | 0.147 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.010 ± 2.376 | 0.011 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.007 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.045 ± 0.001 | 0.041 ± 0.000 | 0.029 ± 0.000 | 0.041 ± 0.010 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.008 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 20.9x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 28.8x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 27.3x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.115 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 224.808 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 113.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 27.9x faster than the slowest successful cell (`tenferro-eager`, 113.715 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`tenferro-eager`, 419.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 13.0x faster than the slowest successful cell (`tenferro-trace`, 367.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 60.6x faster than the slowest successful cell (`tenferro-eager`, 178.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 72.0x faster than the slowest successful cell (`tenferro-eager`, 178.809 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 81.2x faster than the slowest successful cell (`tenferro-trace`, 258.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 64.6x faster than the slowest successful cell (`tenferro-trace`, 258.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 25.7x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 27.7x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`jax-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 33.7x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 25.8x faster than the slowest successful cell (`tenferro-eager`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.5x faster than the slowest successful cell (`tenferro-eager`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.222 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 65.8x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 0.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 60.2x faster than the slowest successful cell (`tenferro-eager`, 0.586 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.586 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 37.0x faster than the slowest successful cell (`tenferro-eager`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 88.3x faster than the slowest successful cell (`tenferro-eager`, 0.774 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`tenferro-eager`, 0.774 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 76.1x faster than the slowest successful cell (`tenferro-eager`, 0.782 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.782 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 51.8x faster than the slowest successful cell (`tenferro-eager`, 0.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 0.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.2x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.1x faster than the slowest successful cell (`pytorch-cpu`, 0.273 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.6x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 28.7x faster than the slowest successful cell (`pytorch-cpu`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.1x faster than the slowest successful cell (`tenferro-eager`, 0.207 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

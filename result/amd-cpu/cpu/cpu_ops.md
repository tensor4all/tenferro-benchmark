# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261113_180725/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_180725/run.yaml`
- Timestamp: `2026-11-13T18:07:25.705019+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261113_180725`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261113_180725/cpu_ops_t1_20261113_180725.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.046 ± 0.000 | 0.045 ± 0.000 | 0.041 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.039 ± 0.000 | 0.042 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.143 ± 0.000 | 0.143 ± 0.003 | 0.131 ± 0.001 | 0.128 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.000 | 0.177 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.186 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.150 ± 0.000 | 0.151 ± 0.000 | 0.177 ± 0.003 | 0.232 ± 0.039 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.000 | 0.190 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | 0.187 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.000 | 0.019 ± 0.000 | 0.020 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.042 ± 0.001 | 0.041 ± 0.000 | 0.039 ± 0.001 | 0.059 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.015 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.001 | 0.039 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.020 ± 0.001 | 0.018 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.026 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.002 | 0.035 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.087 ± 0.000 | 0.086 ± 0.000 | 0.078 ± 0.002 | 0.093 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.081 ± 0.001 | 0.079 ± 0.001 | 0.075 ± 0.001 | 0.075 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.001 | 0.299 ± 0.002 | 0.260 ± 0.002 | 0.250 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.083 ± 0.001 | 0.018 ± 0.000 | 0.140 ± 0.001 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.099 ± 0.001 | 0.034 ± 0.000 | 0.144 ± 0.003 | 0.018 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.086 ± 0.002 | 0.019 ± 0.000 | 0.146 ± 0.003 | 0.016 ± 0.006 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.109 ± 0.001 | 0.040 ± 0.000 | 0.170 ± 0.002 | 0.053 ± 0.020 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.124 ± 0.001 | 0.047 ± 0.000 | 0.132 ± 0.002 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.140 ± 0.041 | 0.074 ± 0.000 | 0.142 ± 0.004 | 0.040 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.125 ± 0.007 | 0.050 ± 0.000 | 0.135 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.143 ± 0.011 | 0.088 ± 0.000 | 0.151 ± 0.004 | 0.055 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.298 ± 0.003 | 0.300 ± 0.002 | 0.322 ± 0.001 | 0.391 ± 0.011 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 16.822 ± 0.116 | 11.497 ± 0.043 | 10.184 ± 0.113 | 8.915 ± 0.017 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 106.028 ± 0.942 | 75.075 ± 0.532 | 65.360 ± 0.290 | 57.118 ± 0.353 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.346 ± 0.046 | 11.554 ± 0.026 | 8.444 ± 0.036 | 8.899 ± 0.044 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 59.931 ± 0.807 | 75.796 ± 0.856 | 51.259 ± 0.556 | 56.758 ± 0.562 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 10.760 ± 0.124 | 5.622 ± 0.051 | 12.114 ± 0.054 | 4.274 ± 0.048 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 74.106 ± 1.428 | 38.334 ± 0.076 | 79.148 ± 0.771 | 30.755 ± 0.511 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 9.906 ± 0.066 | 5.871 ± 0.017 | 5.362 ± 0.090 | 4.278 ± 0.012 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 70.277 ± 0.324 | 39.186 ± 1.080 | 42.702 ± 0.256 | 32.080 ± 0.143 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.034 ± 0.001 | 0.025 ± 0.001 | 0.032 ± 0.001 | 0.040 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.169 ± 0.003 | 0.072 ± 0.000 | 0.142 ± 0.005 | 0.084 ± 0.004 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 16.258 ± 0.050 | 10.009 ± 0.103 | 7.215 ± 0.018 | 7.104 ± 0.057 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 115.036 ± 0.685 | 72.578 ± 0.592 | 50.909 ± 0.223 | 45.358 ± 2.272 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 12.437 ± 0.043 | 9.912 ± 0.066 | 7.223 ± 0.057 | 7.118 ± 0.027 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 90.339 ± 0.484 | 69.210 ± 0.785 | 51.096 ± 0.557 | 53.361 ± 0.347 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.167 ± 0.018 | 0.146 ± 0.001 | 0.168 ± 0.004 | 0.079 ± 0.024 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.824 ± 0.039 | 1.407 ± 0.013 | 1.252 ± 0.017 | 0.658 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 8.525 ± 0.133 | 7.914 ± 0.029 | 5.445 ± 0.107 | 4.086 ± 0.025 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.866 ± 0.009 | 1.403 ± 0.025 | 2.119 ± 0.015 | 0.665 ± 0.004 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.431 ± 0.083 | 7.889 ± 0.056 | 9.372 ± 0.098 | 4.003 ± 0.043 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.774 ± 0.004 | 0.874 ± 0.028 | 0.775 ± 0.007 | 0.655 ± 0.014 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 28.669 ± 0.339 | 19.723 ± 0.136 | 20.555 ± 0.274 | 15.373 ± 0.805 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 192.025 ± 1.349 | 141.119 ± 0.165 | 133.788 ± 0.693 | 108.117 ± 0.901 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 14.897 ± 0.065 | 19.682 ± 0.079 | 15.832 ± 0.164 | 15.377 ± 0.033 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 101.633 ± 0.401 | 140.025 ± 0.646 | 106.551 ± 0.595 | 114.858 ± 0.978 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.125 ± 0.002 | 0.126 ± 0.001 | 0.198 ± 0.003 | 0.186 ± 0.004 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.818 ± 0.025 | 0.812 ± 0.004 | 1.087 ± 0.008 | 1.230 ± 0.030 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.234 ± 0.065 | 3.230 ± 0.034 | 3.460 ± 0.064 | 4.981 ± 0.179 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.093 ± 0.001 | 0.091 ± 0.001 | 0.112 ± 0.002 | 0.107 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.034 ± 0.000 | 0.037 ± 0.000 | 0.051 ± 0.001 | 0.064 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.046 ± 0.002 | 0.049 ± 0.002 | 0.067 ± 0.003 | 0.065 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.080 ± 0.003 | 0.085 ± 0.001 | 0.122 ± 0.003 | 0.157 ± 0.001 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.562 ± 0.003 | 0.572 ± 0.010 | 0.631 ± 0.002 | 0.592 ± 0.006 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.004 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.173 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.175 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.187 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.250 ± 0.002 | 0.036 ± 0.001 | 0.191 ± 0.004 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.265 ± 0.001 | 0.043 ± 0.001 | 0.196 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.278 ± 0.002 | 0.051 ± 0.000 | 0.203 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.099 ± 0.001 | 0.038 ± 0.001 | 0.192 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.108 ± 0.002 | 0.046 ± 0.001 | 0.196 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.114 ± 0.002 | 0.052 ± 0.001 | 0.202 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.688 ± 0.012 | 0.035 ± 0.000 | 0.269 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.696 ± 0.008 | 0.036 ± 0.000 | 0.266 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.696 ± 0.005 | 0.037 ± 0.000 | 0.272 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.188 ± 0.001 | 0.034 ± 0.001 | 0.253 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.190 ± 0.002 | 0.034 ± 0.000 | 0.253 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.195 ± 0.003 | 0.036 ± 0.000 | 0.257 ± 0.005 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.074 ± 0.001 | 0.012 ± 0.000 | 0.070 ± 0.001 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.074 ± 0.016 | 0.012 ± 0.000 | 0.072 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.075 ± 0.001 | 0.012 ± 0.000 | 0.073 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.946 ± 0.027 | 0.059 ± 0.000 | 0.213 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.948 ± 0.006 | 0.061 ± 0.001 | 0.212 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.948 ± 0.010 | 0.065 ± 0.000 | 0.216 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.180 ± 0.001 | 0.062 ± 0.001 | 0.242 ± 0.005 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.183 ± 0.001 | 0.065 ± 0.000 | 0.244 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.189 ± 0.000 | 0.069 ± 0.000 | 0.246 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.113 ± 0.005 | 0.037 ± 0.000 | 0.116 ± 0.002 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.114 ± 0.002 | 0.038 ± 0.000 | 0.120 ± 0.001 | 0.018 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.117 ± 0.003 | 0.040 ± 0.001 | 0.120 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.186 ± 0.001 | 0.026 ± 0.000 | 0.403 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.189 ± 0.002 | 0.027 ± 0.000 | 0.404 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.190 ± 0.002 | 0.028 ± 0.001 | 0.403 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.094 ± 0.001 | 0.026 ± 0.000 | 0.240 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.095 ± 0.002 | 0.027 ± 0.000 | 0.241 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.098 ± 0.002 | 0.028 ± 0.000 | 0.241 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.107 ± 0.005 | 0.022 ± 0.000 | 0.106 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.114 ± 0.002 | 0.031 ± 0.000 | 0.114 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.126 ± 0.003 | 0.045 ± 0.004 | 0.125 ± 0.002 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.223 ± 0.002 | 0.023 ± 0.000 | 0.256 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.239 ± 0.001 | 0.032 ± 0.000 | 0.263 ± 0.003 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.262 ± 0.009 | 0.046 ± 0.006 | 0.278 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.092 ± 0.002 | 0.023 ± 0.000 | 0.208 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.100 ± 0.001 | 0.031 ± 0.000 | 0.215 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.112 ± 0.001 | 0.045 ± 0.000 | 0.225 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.040 ± 0.001 | 0.049 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.024 ± 0.000 | 0.026 ± 0.000 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 35.8x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 32.3x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 34.3x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 51.7x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`jax-cpu`, 0.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 51.8x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 22.6x faster than the slowest successful cell (`jax-cpu`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 54.1x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.7x faster than the slowest successful cell (`jax-cpu`, 0.187 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 0.250 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 0.265 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.192 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 53.3x faster than the slowest successful cell (`tenferro-eager`, 0.688 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 0.688 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 52.7x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 47.7x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.7x faster than the slowest successful cell (`tenferro-eager`, 0.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 73.9x faster than the slowest successful cell (`tenferro-eager`, 0.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 69.6x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 55.2x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 31.1x faster than the slowest successful cell (`pytorch-cpu`, 0.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.404 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.240 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.278 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.215 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.225 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261113_184804/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261113_184804/run.yaml`
- Timestamp: `2026-11-13T18:48:04.785108+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261113_184804`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261113_184804/cpu_ops_t4_20261113_184804.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.018 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.045 ± 0.000 | 0.044 ± 0.000 | 0.039 ± 0.000 | 0.054 ± 0.012 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.039 ± 0.001 | 0.044 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.142 ± 0.001 | 0.140 ± 0.000 | 0.131 ± 0.003 | 0.146 ± 0.032 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.177 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.001 | 0.016 ± 0.000 | 0.226 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.153 ± 0.000 | 0.155 ± 0.001 | 0.122 ± 0.003 | 0.269 ± 0.094 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.189 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.025 ± 0.000 | 0.191 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.023 ± 0.001 | 0.025 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.041 ± 0.000 | 0.059 ± 0.001 | 0.065 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.036 ± 0.001 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.032 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.001 | 0.020 ± 0.000 | 0.040 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.027 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.084 ± 0.001 | 0.084 ± 0.001 | 0.074 ± 0.002 | 0.092 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.080 ± 0.002 | 0.079 ± 0.000 | 0.073 ± 0.001 | 0.076 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.297 ± 0.002 | 0.296 ± 0.001 | 0.258 ± 0.004 | 0.250 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.093 ± 0.012 | 0.020 ± 0.000 | 0.141 ± 0.002 | 0.012 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.105 ± 0.009 | 0.037 ± 0.000 | 0.144 ± 0.001 | 0.013 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.091 ± 0.002 | 0.022 ± 0.000 | 0.146 ± 0.001 | 0.021 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.116 ± 0.017 | 0.042 ± 0.001 | 0.170 ± 0.001 | 0.079 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.132 ± 0.000 | 0.058 ± 0.000 | 0.173 ± 0.001 | 0.024 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.143 ± 0.008 | 0.084 ± 0.001 | 0.184 ± 0.002 | 0.080 ± 0.011 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.135 ± 0.001 | 0.061 ± 0.000 | 0.177 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.151 ± 0.016 | 0.098 ± 0.000 | 0.196 ± 0.002 | 0.094 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.519 ± 0.002 | 0.518 ± 0.001 | 0.339 ± 0.004 | 0.517 ± 0.018 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 31.288 ± 0.248 | 26.233 ± 0.054 | 7.136 ± 6.316 | 10.474 ± 0.989 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 152.667 ± 0.591 | 116.296 ± 0.407 | 42.654 ± 1.450 | 57.033 ± 5.921 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 18.559 ± 0.302 | 26.072 ± 0.150 | 5.913 ± 0.192 | 10.033 ± 0.857 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 83.913 ± 0.425 | 116.946 ± 0.512 | 30.933 ± 8.038 | 56.753 ± 3.381 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 16.308 ± 0.275 | 9.288 ± 0.018 | 8.886 ± 0.360 | 6.833 ± 1.812 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 332.526 ± 30.283 | 309.134 ± 35.759 | 46.945 ± 0.688 | 30.970 ± 5.157 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 14.337 ± 0.271 | 9.853 ± 0.046 | 3.588 ± 0.045 | 5.805 ± 0.713 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 204.774 ± 27.075 | 323.223 ± 32.445 | 24.040 ± 0.313 | 33.606 ± 8.323 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.068 ± 0.002 | 0.055 ± 0.000 | 0.034 ± 0.004 | 0.038 ± 0.004 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.252 ± 0.024 | 0.157 ± 0.001 | 0.174 ± 0.007 | 0.079 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 169.854 ± 107.086 | 17.660 ± 35.381 | 22.049 ± 3.456 | 8.684 ± 1.591 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 433.802 ± 14.493 | 373.132 ± 163.877 | 33.498 ± 0.444 | 53.328 ± 9.748 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 44.041 ± 48.751 | 15.280 ± 11.350 | 5.190 ± 0.073 | 10.464 ± 2.196 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 264.197 ± 12.897 | 382.293 ± 14.663 | 32.454 ± 0.389 | 55.912 ± 11.714 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.174 ± 0.022 | 0.158 ± 0.001 | 0.170 ± 0.005 | 0.064 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 3.065 ± 0.004 | 2.623 ± 0.011 | 3.018 ± 2.977 | 0.992 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 279.107 ± 20.002 | 311.145 ± 24.081 | 2.934 ± 0.161 | 3.740 ± 0.163 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.495 ± 0.018 | 2.614 ± 0.008 | 1.444 ± 0.077 | 0.964 ± 0.099 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 124.857 ± 17.964 | 304.917 ± 20.887 | 4.783 ± 0.112 | 3.686 ± 0.017 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.338 ± 0.005 | 1.496 ± 0.001 | 0.985 ± 0.032 | 0.897 ± 0.022 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 35.600 ± 0.358 | 24.588 ± 0.145 | 66.351 ± 15.872 | 18.306 ± 0.689 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 237.262 ± 0.946 | 174.782 ± 1.639 | 92.135 ± 1.412 | 100.959 ± 3.761 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 20.405 ± 0.192 | 26.208 ± 0.097 | 56.047 ± 14.996 | 17.038 ± 1.250 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 125.724 ± 1.340 | 176.963 ± 1.671 | 81.293 ± 0.909 | 101.957 ± 5.852 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.178 ± 0.011 | 0.172 ± 0.001 | 0.149 ± 0.008 | 0.170 ± 0.045 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.918 ± 0.002 | 0.919 ± 0.007 | 0.684 ± 0.038 | 0.716 ± 0.090 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 3.652 ± 0.029 | 3.562 ± 0.013 | 2.489 ± 0.168 | 2.752 ± 0.228 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.277 ± 0.002 | 0.278 ± 0.002 | 0.175 ± 0.001 | 0.115 ± 0.027 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.035 ± 0.000 | 0.039 ± 0.001 | 0.050 ± 0.002 | 0.083 ± 0.003 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.076 ± 0.002 | 0.082 ± 0.005 | 0.093 ± 0.004 | 0.153 ± 0.023 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.110 ± 0.005 | 0.117 ± 0.017 | 0.133 ± 0.003 | 0.192 ± 0.003 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 1.090 ± 0.006 | 1.082 ± 0.006 | 0.825 ± 0.169 | 0.828 ± 0.028 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.174 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.178 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.189 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.262 ± 0.004 | 0.044 ± 0.000 | 0.192 ± 0.004 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.274 ± 0.003 | 0.052 ± 0.000 | 0.196 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.286 ± 0.005 | 0.058 ± 0.001 | 0.202 ± 0.003 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.109 ± 0.002 | 0.046 ± 0.000 | 0.189 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.118 ± 0.002 | 0.052 ± 0.000 | 0.196 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.123 ± 0.002 | 0.059 ± 0.001 | 0.201 ± 0.004 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.729 ± 0.006 | 0.045 ± 0.000 | 0.372 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.723 ± 0.008 | 0.046 ± 0.001 | 0.369 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.733 ± 0.006 | 0.047 ± 0.000 | 0.371 ± 0.004 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.217 ± 0.002 | 0.042 ± 0.000 | 0.317 ± 0.018 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.219 ± 0.002 | 0.043 ± 0.000 | 0.325 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.224 ± 0.005 | 0.044 ± 0.000 | 0.327 ± 0.008 | 0.018 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.081 ± 0.003 | 0.014 ± 0.000 | 0.070 ± 0.002 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.081 ± 0.009 | 0.015 ± 0.000 | 0.072 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.081 ± 0.000 | 0.015 ± 0.000 | 0.073 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.969 ± 0.008 | 0.074 ± 0.000 | 0.281 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.977 ± 0.002 | 0.077 ± 0.000 | 0.279 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.977 ± 0.015 | 0.081 ± 0.001 | 0.281 ± 0.005 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.207 ± 0.001 | 0.077 ± 0.001 | 0.303 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.211 ± 0.003 | 0.079 ± 0.001 | 0.310 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.214 ± 0.002 | 0.083 ± 0.002 | 0.312 ± 0.007 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.124 ± 0.006 | 0.048 ± 0.001 | 0.117 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.125 ± 0.012 | 0.048 ± 0.000 | 0.119 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.125 ± 0.002 | 0.050 ± 0.000 | 0.120 ± 0.002 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.195 ± 0.004 | 0.033 ± 0.000 | 0.400 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.198 ± 0.004 | 0.034 ± 0.000 | 0.399 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.198 ± 0.003 | 0.035 ± 0.000 | 0.398 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.104 ± 0.002 | 0.033 ± 0.000 | 0.239 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.106 ± 0.001 | 0.034 ± 0.001 | 0.242 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.106 ± 0.001 | 0.035 ± 0.000 | 0.241 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.113 ± 0.002 | 0.027 ± 0.000 | 0.104 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.121 ± 0.018 | 0.035 ± 0.001 | 0.112 ± 0.001 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.132 ± 0.010 | 0.050 ± 0.001 | 0.123 ± 0.000 | 0.027 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.231 ± 0.002 | 0.029 ± 0.000 | 0.259 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.246 ± 0.002 | 0.037 ± 0.000 | 0.263 ± 0.006 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.271 ± 0.003 | 0.051 ± 0.001 | 0.278 ± 0.002 | 0.033 ± 0.006 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.099 ± 0.001 | 0.027 ± 0.000 | 0.208 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.107 ± 0.001 | 0.036 ± 0.000 | 0.214 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.119 ± 0.001 | 0.050 ± 0.000 | 0.227 ± 0.009 | 0.033 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.029 ± 0.000 | 0.030 ± 0.000 | 0.019 ± 1.439 | 0.013 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.051 ± 0.001 | 0.050 ± 0.001 | 0.034 ± 0.001 | 0.051 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.020 ± 0.001 | 0.019 ± 0.000 | 0.023 ± 0.000 | 0.027 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 32.2x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 28.9x faster than the slowest successful cell (`jax-cpu`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 31.3x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.4x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-eager`, 332.526 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 323.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 19.6x faster than the slowest successful cell (`tenferro-eager`, 169.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 433.802 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 11.8x faster than the slowest successful cell (`tenferro-trace`, 382.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 83.2x faster than the slowest successful cell (`tenferro-trace`, 311.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 106.1x faster than the slowest successful cell (`tenferro-trace`, 311.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 82.7x faster than the slowest successful cell (`tenferro-trace`, 304.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 63.8x faster than the slowest successful cell (`tenferro-trace`, 304.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 44.4x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 45.2x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 46.3x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.6x faster than the slowest successful cell (`tenferro-eager`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 0.286 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.189 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 55.1x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.1x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 48.5x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.723 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 35.9x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`tenferro-eager`, 0.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 0.317 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 73.7x faster than the slowest successful cell (`tenferro-eager`, 0.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 0.969 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.3x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 45.4x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.977 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.303 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 0.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 30.7x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.400 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 29.0x faster than the slowest successful cell (`pytorch-cpu`, 0.399 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.399 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.398 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.239 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.242 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.3x faster than the slowest successful cell (`pytorch-cpu`, 0.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.5x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.208 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

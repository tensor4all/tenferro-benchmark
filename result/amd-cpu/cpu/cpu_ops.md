# CPU Benchmark Results

- Target profile: `amd-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/amd-cpu/cpu/einsum/20261115_180744/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_180744/run.yaml`
- Timestamp: `2026-11-15T18:07:44.075207+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261115_180744`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261115_180744/cpu_ops_t1_20261115_180744.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.001 | 0.046 ± 0.001 | 0.042 ± 0.001 | 0.032 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.042 ± 0.000 | 0.044 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.146 ± 0.001 | 0.145 ± 0.002 | 0.135 ± 0.001 | 0.147 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.001 | 0.180 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.000 | 0.191 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.168 ± 0.002 | 0.168 ± 0.000 | 0.178 ± 0.001 | 0.204 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.017 ± 0.000 | 0.216 ± 0.033 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.000 | 0.218 ± 0.035 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.021 ± 0.000 | 0.030 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.042 ± 0.000 | 0.040 ± 0.001 | 0.079 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.001 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.025 ± 0.000 | 0.024 ± 0.017 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.001 | 0.018 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.001 | 0.019 ± 0.001 | 0.028 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.027 ± 0.001 | 0.031 ± 0.001 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.092 ± 0.002 | 0.090 ± 0.000 | 0.082 ± 0.002 | 0.098 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.082 ± 0.001 | 0.082 ± 0.002 | 0.076 ± 0.003 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.308 ± 0.004 | 0.304 ± 0.001 | 0.267 ± 0.012 | 0.255 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.080 ± 0.002 | 0.018 ± 0.000 | 0.141 ± 0.002 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.099 ± 0.004 | 0.035 ± 0.000 | 0.144 ± 0.001 | 0.015 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.083 ± 0.001 | 0.020 ± 0.002 | 0.148 ± 0.004 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.108 ± 0.001 | 0.041 ± 0.000 | 0.174 ± 0.004 | 0.035 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.120 ± 0.001 | 0.047 ± 0.000 | 0.139 ± 0.007 | 0.025 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.133 ± 0.004 | 0.073 ± 0.000 | 0.144 ± 0.003 | 0.042 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.127 ± 0.005 | 0.050 ± 0.000 | 0.139 ± 0.003 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.140 ± 0.016 | 0.088 ± 0.000 | 0.155 ± 0.002 | 0.056 ± 0.001 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.298 ± 0.000 | 0.301 ± 0.005 | 0.324 ± 0.004 | 0.399 ± 0.002 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 17.079 ± 0.083 | 11.701 ± 0.071 | 10.363 ± 0.170 | 8.987 ± 0.152 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 107.736 ± 0.925 | 75.499 ± 0.528 | 66.714 ± 0.198 | 57.633 ± 1.333 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 9.551 ± 0.094 | 11.792 ± 0.197 | 8.498 ± 0.089 | 9.073 ± 0.024 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 60.881 ± 0.431 | 75.751 ± 0.129 | 52.871 ± 2.890 | 56.907 ± 0.733 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 11.006 ± 0.049 | 5.729 ± 0.095 | 12.668 ± 0.321 | 4.455 ± 0.089 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 74.355 ± 0.526 | 39.187 ± 0.337 | 81.228 ± 3.709 | 32.796 ± 1.484 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 10.175 ± 0.134 | 5.986 ± 0.083 | 5.517 ± 0.107 | 4.460 ± 0.033 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 71.228 ± 1.391 | 40.230 ± 1.081 | 42.739 ± 0.292 | 32.734 ± 0.838 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.034 ± 0.002 | 0.025 ± 0.000 | 0.034 ± 0.001 | 0.037 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.152 ± 0.039 | 0.073 ± 0.001 | 0.154 ± 0.003 | 0.080 ± 0.003 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 16.678 ± 0.077 | 10.253 ± 0.315 | 7.317 ± 0.079 | 7.205 ± 0.051 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 119.255 ± 0.544 | 72.784 ± 0.096 | 53.224 ± 0.589 | 48.953 ± 0.463 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 12.700 ± 0.148 | 10.042 ± 0.176 | 7.310 ± 0.087 | 7.322 ± 0.114 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 93.841 ± 0.615 | 69.156 ± 0.817 | 52.166 ± 0.971 | 51.692 ± 2.203 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.161 ± 0.008 | 0.145 ± 0.001 | 0.175 ± 0.004 | 0.079 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.860 ± 0.018 | 1.402 ± 0.008 | 1.241 ± 0.019 | 0.677 ± 0.017 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 8.836 ± 0.173 | 8.195 ± 0.093 | 5.604 ± 0.043 | 4.139 ± 0.082 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.883 ± 0.016 | 1.409 ± 0.003 | 1.734 ± 0.034 | 0.686 ± 0.016 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 4.805 ± 0.216 | 8.062 ± 0.103 | 9.522 ± 0.042 | 4.138 ± 0.059 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.779 ± 0.054 | 0.855 ± 0.006 | 0.779 ± 0.006 | 0.673 ± 0.020 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 29.205 ± 0.343 | 20.323 ± 0.318 | 18.979 ± 0.077 | 15.875 ± 0.282 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 191.679 ± 3.484 | 137.302 ± 0.553 | 136.092 ± 1.364 | 107.898 ± 1.395 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 15.130 ± 0.135 | 20.227 ± 0.184 | 15.936 ± 0.056 | 15.856 ± 0.140 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 102.092 ± 1.339 | 136.095 ± 1.266 | 108.490 ± 0.377 | 113.814 ± 1.172 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.130 ± 0.002 | 0.129 ± 0.001 | 0.200 ± 0.007 | 0.198 ± 0.004 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.860 ± 0.051 | 0.846 ± 0.008 | 1.096 ± 0.020 | 1.469 ± 0.055 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 3.253 ± 0.029 | 3.297 ± 0.054 | 3.696 ± 0.099 | 4.716 ± 0.052 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.092 ± 0.001 | 0.091 ± 0.001 | 0.115 ± 0.003 | 0.122 ± 0.002 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.035 ± 0.002 | 0.037 ± 0.000 | 0.053 ± 0.001 | 0.047 ± 0.001 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.046 ± 0.004 | 0.049 ± 0.002 | 0.073 ± 0.009 | 0.084 ± 0.006 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.081 ± 0.000 | 0.086 ± 0.002 | 0.121 ± 0.002 | 0.154 ± 0.029 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.568 ± 0.010 | 0.562 ± 0.004 | 0.635 ± 0.005 | 0.603 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.179 ± 0.018 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.001 | 0.008 ± 0.000 | 0.013 ± 0.000 | 0.180 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.202 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.253 ± 0.006 | 0.035 ± 0.001 | 0.196 ± 0.009 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.261 ± 0.010 | 0.041 ± 0.000 | 0.200 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.269 ± 0.003 | 0.047 ± 0.000 | 0.205 ± 0.005 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.096 ± 0.001 | 0.036 ± 0.000 | 0.196 ± 0.010 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.105 ± 0.001 | 0.043 ± 0.001 | 0.200 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.108 ± 0.002 | 0.049 ± 0.000 | 0.202 ± 0.004 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.703 ± 0.010 | 0.035 ± 0.000 | 0.278 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.699 ± 0.018 | 0.036 ± 0.000 | 0.279 ± 0.008 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.694 ± 0.020 | 0.037 ± 0.000 | 0.276 ± 0.005 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.186 ± 0.006 | 0.033 ± 0.000 | 0.257 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.188 ± 0.003 | 0.034 ± 0.000 | 0.257 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.189 ± 0.001 | 0.035 ± 0.000 | 0.258 ± 0.005 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.072 ± 0.001 | 0.012 ± 0.000 | 0.070 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.073 ± 0.003 | 0.012 ± 0.000 | 0.073 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.073 ± 0.000 | 0.012 ± 0.000 | 0.074 ± 0.002 | 0.017 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.939 ± 0.016 | 0.059 ± 0.001 | 0.218 ± 0.011 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.967 ± 0.016 | 0.061 ± 0.000 | 0.217 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.956 ± 0.007 | 0.064 ± 0.000 | 0.220 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.177 ± 0.003 | 0.063 ± 0.001 | 0.248 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.181 ± 0.004 | 0.065 ± 0.001 | 0.249 ± 0.006 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.183 ± 0.004 | 0.068 ± 0.000 | 0.251 ± 0.007 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.135 ± 0.008 | 0.038 ± 0.000 | 0.118 ± 0.001 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.115 ± 0.011 | 0.037 ± 0.000 | 0.122 ± 0.001 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.115 ± 0.002 | 0.040 ± 0.000 | 0.122 ± 0.001 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.185 ± 0.001 | 0.026 ± 0.000 | 0.413 ± 0.008 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.183 ± 0.009 | 0.026 ± 0.000 | 0.407 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.185 ± 0.002 | 0.027 ± 0.000 | 0.408 ± 0.009 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.094 ± 0.002 | 0.026 ± 0.000 | 0.244 ± 0.004 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.096 ± 0.078 | 0.026 ± 0.000 | 0.244 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.094 ± 0.001 | 0.028 ± 0.000 | 0.243 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.104 ± 0.009 | 0.022 ± 0.000 | 0.108 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.111 ± 0.020 | 0.030 ± 0.001 | 0.115 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.122 ± 0.002 | 0.045 ± 0.001 | 0.126 ± 0.001 | 0.022 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.222 ± 0.004 | 0.023 ± 0.000 | 0.263 ± 0.007 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.240 ± 0.006 | 0.031 ± 0.000 | 0.271 ± 0.003 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.264 ± 0.001 | 0.044 ± 0.000 | 0.283 ± 0.006 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.090 ± 0.002 | 0.023 ± 0.001 | 0.214 ± 0.015 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.099 ± 0.003 | 0.030 ± 0.000 | 0.217 ± 0.004 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.109 ± 0.002 | 0.045 ± 0.001 | 0.227 ± 0.005 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.028 ± 0.001 | 0.022 ± 0.000 | 0.042 ± 0.001 | 0.050 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.012 ± 0.005 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.012 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.021 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.020 ± 0.001 | 0.019 ± 0.000 | 0.024 ± 0.000 | 0.030 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 33.1x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 35.0x faster than the slowest successful cell (`jax-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 34.3x faster than the slowest successful cell (`jax-cpu`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.141 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 43.7x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 44.5x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 45.5x faster than the slowest successful cell (`jax-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 26.0x faster than the slowest successful cell (`jax-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 0.253 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 12.5x faster than the slowest successful cell (`pytorch-cpu`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 53.7x faster than the slowest successful cell (`tenferro-eager`, 0.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`tenferro-eager`, 0.703 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 52.1x faster than the slowest successful cell (`tenferro-eager`, 0.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`tenferro-eager`, 0.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 44.5x faster than the slowest successful cell (`tenferro-eager`, 0.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`tenferro-eager`, 0.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 68.6x faster than the slowest successful cell (`tenferro-eager`, 0.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.939 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 68.5x faster than the slowest successful cell (`tenferro-eager`, 0.967 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 0.967 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 52.1x faster than the slowest successful cell (`tenferro-eager`, 0.956 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.956 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.248 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.251 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 0.413 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 29.8x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.407 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 0.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.271 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.283 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 0.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.217 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/amd-cpu/cpu/einsum/20261115_184936/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/amd-cpu/cpu/einsum/20261115_184936/run.yaml`
- Timestamp: `2026-11-15T18:49:36.321391+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/amd-cpu/cpu/einsum/20261115_184936`.

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


- CSV: `data/results/amd-cpu/cpu/einsum/20261115_184936/cpu_ops_t4_20261115_184936.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.000 | 0.046 ± 0.000 | 0.041 ± 0.000 | 0.046 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.000 | 0.040 ± 0.000 | 0.040 ± 0.001 | 0.044 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.145 ± 0.000 | 0.145 ± 0.002 | 0.133 ± 0.001 | 0.133 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.000 | 0.179 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.227 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.152 ± 0.002 | 0.153 ± 0.001 | 0.123 ± 0.003 | 0.340 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.190 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.026 ± 0.000 | 0.191 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.022 ± 0.000 | 0.020 ± 0.000 | 0.032 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.024 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.043 ± 0.000 | 0.042 ± 0.000 | 0.060 ± 0.001 | 0.065 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.031 ± 0.000 | 0.015 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.000 | 0.018 ± 0.000 | 0.037 ± 0.000 | 0.035 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.033 ± 0.000 | 0.019 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.064 ± 0.065 | 0.051 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.000 | 0.036 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.087 ± 0.000 | 0.078 ± 0.002 | 0.098 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.082 ± 0.001 | 0.080 ± 0.001 | 0.075 ± 0.001 | 0.077 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.303 ± 0.002 | 0.302 ± 0.001 | 0.265 ± 0.004 | 0.254 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.090 ± 0.001 | 0.020 ± 0.000 | 0.143 ± 0.001 | 0.014 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.109 ± 0.004 | 0.037 ± 0.000 | 0.147 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.093 ± 0.001 | 0.022 ± 0.000 | 0.150 ± 0.002 | 0.023 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.118 ± 0.003 | 0.043 ± 0.000 | 0.174 ± 0.003 | 0.084 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.136 ± 0.013 | 0.061 ± 0.001 | 0.174 ± 0.002 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.146 ± 0.005 | 0.088 ± 0.000 | 0.189 ± 0.003 | 0.081 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.137 ± 0.003 | 0.064 ± 0.000 | 0.181 ± 0.001 | 0.030 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.157 ± 0.005 | 0.102 ± 0.000 | 0.201 ± 0.002 | 0.096 ± 0.003 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.527 ± 0.002 | 0.521 ± 0.006 | 0.344 ± 0.013 | 0.512 ± 0.005 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 32.623 ± 0.131 | 26.930 ± 0.094 | 7.165 ± 1.732 | 11.994 ± 1.971 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 155.695 ± 0.423 | 119.868 ± 0.769 | 43.299 ± 2.671 | 56.961 ± 1.285 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 18.774 ± 0.375 | 26.600 ± 0.144 | 5.977 ± 0.070 | 11.035 ± 1.463 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 86.186 ± 0.275 | 119.245 ± 0.219 | 30.709 ± 0.182 | 54.395 ± 9.059 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 16.770 ± 0.404 | 9.487 ± 0.104 | 8.790 ± 0.478 | 6.353 ± 0.716 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 361.531 ± 23.529 | 327.542 ± 9.755 | 50.575 ± 5.425 | 35.012 ± 9.836 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 14.482 ± 0.096 | 9.996 ± 0.038 | 3.593 ± 0.034 | 5.984 ± 1.002 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 214.753 ± 9.797 | 324.988 ± 44.644 | 24.331 ± 0.372 | 30.301 ± 6.882 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.072 ± 0.003 | 0.056 ± 0.001 | 0.037 ± 0.001 | 0.038 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.263 ± 0.016 | 0.158 ± 0.002 | 0.175 ± 0.002 | 0.106 ± 0.004 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 268.663 ± 17.699 | 16.050 ± 59.148 | 4.836 ± 0.041 | 6.561 ± 2.645 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 322.530 ± 154.186 | 391.820 ± 15.321 | 33.579 ± 0.319 | 50.945 ± 4.075 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 66.104 ± 59.636 | 57.343 ± 35.456 | 5.365 ± 0.057 | 7.987 ± 2.990 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 257.304 ± 11.257 | 390.937 ± 2.909 | 33.512 ± 0.524 | 52.166 ± 8.537 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.180 ± 0.029 | 0.153 ± 0.001 | 0.171 ± 0.002 | 0.065 ± 0.003 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 3.103 ± 0.028 | 2.677 ± 0.034 | 1.047 ± 0.009 | 0.894 ± 0.098 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 285.251 ± 9.903 | 313.985 ± 23.988 | 2.933 ± 0.016 | 3.868 ± 0.289 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 1.537 ± 0.036 | 2.642 ± 0.174 | 1.241 ± 0.069 | 0.971 ± 0.060 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 136.895 ± 18.009 | 299.005 ± 12.114 | 4.733 ± 0.071 | 3.835 ± 0.084 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 1.360 ± 0.031 | 1.504 ± 0.003 | 1.004 ± 0.007 | 0.927 ± 0.071 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 37.190 ± 4.087 | 25.418 ± 0.210 | 16.434 ± 0.157 | 16.632 ± 1.302 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 241.224 ± 4.091 | 175.090 ± 0.285 | 93.537 ± 0.588 | 101.027 ± 6.267 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 20.776 ± 0.134 | 26.815 ± 0.349 | 14.735 ± 0.080 | 17.973 ± 0.753 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 127.239 ± 1.427 | 175.813 ± 1.320 | 82.953 ± 1.123 | 191.812 ± 38.114 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.179 ± 0.006 | 0.180 ± 0.002 | 0.152 ± 0.009 | 0.174 ± 0.034 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.968 ± 0.037 | 0.930 ± 0.011 | 0.696 ± 0.040 | 0.846 ± 0.257 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 3.753 ± 0.084 | 3.621 ± 0.041 | 2.620 ± 0.210 | 2.779 ± 0.196 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.285 ± 0.002 | 0.279 ± 0.004 | 0.179 ± 0.003 | 0.117 ± 0.005 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.036 ± 0.001 | 0.039 ± 0.000 | 0.051 ± 0.002 | 0.082 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.076 ± 0.005 | 0.082 ± 0.005 | 0.094 ± 0.004 | 0.153 ± 0.046 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.110 ± 0.004 | 0.118 ± 0.018 | 0.135 ± 0.002 | 0.202 ± 0.017 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 1.113 ± 0.014 | 1.093 ± 0.003 | 0.832 ± 0.018 | 0.847 ± 0.022 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.171 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.178 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.226 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.263 ± 0.001 | 0.044 ± 0.001 | 0.195 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.277 ± 0.002 | 0.051 ± 0.000 | 0.199 ± 0.004 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.287 ± 0.007 | 0.058 ± 0.000 | 0.209 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.111 ± 0.002 | 0.046 ± 0.001 | 0.193 ± 0.004 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.119 ± 0.002 | 0.053 ± 0.001 | 0.201 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.124 ± 0.000 | 0.060 ± 0.001 | 0.204 ± 0.003 | 0.018 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.737 ± 0.010 | 0.045 ± 0.000 | 0.382 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.729 ± 0.004 | 0.046 ± 0.000 | 0.379 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.754 ± 0.001 | 0.047 ± 0.001 | 0.382 ± 0.006 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.220 ± 0.003 | 0.042 ± 0.001 | 0.330 ± 0.010 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.221 ± 0.002 | 0.043 ± 0.001 | 0.332 ± 0.009 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.228 ± 0.002 | 0.045 ± 0.000 | 0.334 ± 0.015 | 0.018 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.081 ± 0.000 | 0.014 ± 0.000 | 0.071 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.085 ± 0.007 | 0.015 ± 0.000 | 0.073 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.082 ± 0.001 | 0.015 ± 0.000 | 0.073 ± 0.001 | 0.014 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.984 ± 0.013 | 0.074 ± 0.000 | 0.287 ± 0.009 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.994 ± 0.010 | 0.076 ± 0.000 | 0.285 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.996 ± 0.005 | 0.080 ± 0.001 | 0.286 ± 0.003 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.211 ± 0.003 | 0.076 ± 0.000 | 0.313 ± 0.013 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.213 ± 0.001 | 0.079 ± 0.002 | 0.313 ± 0.007 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.218 ± 0.002 | 0.083 ± 0.001 | 0.319 ± 0.007 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.125 ± 0.006 | 0.047 ± 0.001 | 0.119 ± 0.003 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.128 ± 0.005 | 0.048 ± 0.000 | 0.121 ± 0.002 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.127 ± 0.001 | 0.050 ± 0.001 | 0.122 ± 0.003 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.197 ± 0.001 | 0.033 ± 0.001 | 0.408 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.198 ± 0.003 | 0.034 ± 0.000 | 0.408 ± 0.004 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.200 ± 0.001 | 0.035 ± 0.000 | 0.416 ± 0.010 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.106 ± 0.001 | 0.033 ± 0.000 | 0.246 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.106 ± 0.001 | 0.034 ± 0.001 | 0.246 ± 0.034 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.108 ± 0.001 | 0.035 ± 0.000 | 0.243 ± 0.005 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.112 ± 0.001 | 0.028 ± 0.001 | 0.106 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.123 ± 0.002 | 0.036 ± 0.000 | 0.114 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.133 ± 0.001 | 0.051 ± 0.001 | 0.125 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.234 ± 0.005 | 0.029 ± 0.000 | 0.267 ± 0.145 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.250 ± 0.004 | 0.037 ± 0.000 | 0.269 ± 0.007 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.272 ± 0.002 | 0.051 ± 0.001 | 0.289 ± 0.008 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.099 ± 0.001 | 0.028 ± 0.000 | 0.212 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.108 ± 0.001 | 0.037 ± 0.000 | 0.218 ± 0.004 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.120 ± 0.000 | 0.052 ± 0.001 | 0.231 ± 0.006 | 0.027 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.032 ± 0.000 | 0.031 ± 0.000 | 11.140 ± 0.981 | 0.014 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.054 ± 0.001 | 0.050 ± 0.001 | 0.210 ± 0.280 | 0.036 ± 0.005 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.002 ± 0.000 | 0.015 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.024 ± 0.000 | 0.027 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.227 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 30.1x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`jax-cpu`, 0.190 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.191 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.143 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_jvp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 361.531 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `jax-cpu` is 10.7x faster than the slowest successful cell (`tenferro-trace`, 324.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_lu_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 324.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `jax-cpu` is 41.0x faster than the slowest successful cell (`tenferro-eager`, 268.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 55.5x faster than the slowest successful cell (`tenferro-eager`, 268.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`tenferro-eager`, 268.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_jvp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 391.820 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 66.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_qr_vjp` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`tenferro-trace`, 390.937 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 81.2x faster than the slowest successful cell (`tenferro-trace`, 313.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_jvp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 107.1x faster than the slowest successful cell (`tenferro-trace`, 313.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `jax-cpu` is 78.0x faster than the slowest successful cell (`tenferro-trace`, 299.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_vjp` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 63.2x faster than the slowest successful cell (`tenferro-trace`, 299.005 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 42.6x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 43.4x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 49.6x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`jax-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 0.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 15.0x faster than the slowest successful cell (`tenferro-eager`, 0.287 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.204 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 55.4x faster than the slowest successful cell (`tenferro-eager`, 0.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 49.1x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.729 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 35.4x faster than the slowest successful cell (`tenferro-eager`, 0.754 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.754 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.9x faster than the slowest successful cell (`pytorch-cpu`, 0.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 73.1x faster than the slowest successful cell (`tenferro-eager`, 0.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`tenferro-eager`, 0.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 64.0x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`tenferro-eager`, 0.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 46.1x faster than the slowest successful cell (`tenferro-eager`, 0.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 24.6x faster than the slowest successful cell (`pytorch-cpu`, 0.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.313 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.3x faster than the slowest successful cell (`pytorch-cpu`, 0.319 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 30.8x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.408 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 29.1x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.416 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 16.5x faster than the slowest successful cell (`pytorch-cpu`, 0.246 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 19.4x faster than the slowest successful cell (`pytorch-cpu`, 0.267 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.269 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.289 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`32x32`): `jax-cpu` is 786.6x faster than the slowest successful cell (`pytorch-cpu`, 11.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 348.9x faster than the slowest successful cell (`pytorch-cpu`, 11.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 358.1x faster than the slowest successful cell (`pytorch-cpu`, 11.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

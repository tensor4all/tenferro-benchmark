# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `9e383937df4e365bd68bf5b16dbfcaaa269bcc82`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261115_172247/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_172247/run.yaml`
- Timestamp: `2026-11-15T17:22:47.463268+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261115_172247`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261115_172247/cpu_ops_t1_20261115_172247.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.002 | 0.011 ± 0.002 | 0.013 ± 0.005 | 0.018 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.006 | 0.018 ± 0.002 | 0.027 ± 0.022 | 0.043 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.037 ± 0.007 | 0.028 ± 0.017 | 0.045 ± 0.031 | 0.049 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.117 ± 0.061 | 0.125 ± 0.040 | 0.094 ± 0.011 | 0.113 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.002 | 0.004 ± 0.001 | 0.019 ± 0.003 | 0.149 ± 0.079 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.002 | 0.005 ± 0.002 | 0.011 ± 0.007 | 0.126 ± 0.057 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.068 ± 0.037 | 0.059 ± 0.001 | 0.042 ± 0.034 | 0.110 ± 0.035 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.005 ± 0.002 | 0.009 ± 0.001 | 0.107 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.009 ± 0.006 | 0.007 ± 0.001 | 0.014 ± 0.002 | 0.085 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.005 | 0.012 ± 0.003 | 0.012 ± 0.008 | 0.015 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.018 ± 0.006 | 0.022 ± 0.004 | 0.028 ± 0.008 | 0.030 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.020 ± 0.005 | 0.014 ± 0.003 | 0.020 ± 0.012 | 0.020 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.019 | 0.051 ± 0.005 | 0.048 ± 0.004 | 0.051 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.002 | 0.009 ± 0.002 | 0.012 ± 0.001 | 0.009 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.016 ± 0.008 | 0.012 ± 0.007 | 0.019 ± 0.010 | 0.015 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.004 | 0.008 ± 0.003 | 0.015 ± 0.001 | 0.012 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.022 ± 0.022 | 0.030 ± 0.012 | 0.022 ± 0.006 | 0.022 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.016 ± 0.003 | 0.013 ± 0.001 | 0.026 ± 0.009 | 0.067 ± 0.020 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.062 ± 0.043 | 0.049 ± 0.008 | 0.043 ± 0.015 | 0.156 ± 0.067 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.069 ± 0.012 | 0.072 ± 0.020 | 0.097 ± 0.103 | 0.086 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.291 ± 0.055 | 0.213 ± 0.071 | 0.211 ± 0.024 | 0.238 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.034 ± 0.001 | 0.014 ± 0.010 | 0.063 ± 0.003 | 0.010 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.074 ± 0.017 | 0.017 ± 0.001 | 0.074 ± 0.051 | 0.009 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.041 ± 0.003 | 0.015 ± 0.009 | 0.077 ± 0.033 | 0.015 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.000 | 0.025 ± 0.010 | 0.082 ± 0.011 | 0.031 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.046 ± 0.001 | 0.031 ± 0.014 | 0.078 ± 0.032 | 0.016 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.064 ± 0.043 | 0.060 ± 0.009 | 0.067 ± 0.050 | 0.023 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.153 ± 0.018 | 0.038 ± 0.003 | 0.070 ± 0.028 | 0.022 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.066 ± 0.002 | 0.070 ± 0.021 | 0.100 ± 0.026 | 0.028 ± 0.006 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.324 ± 0.001 | 0.372 ± 0.020 | 0.393 ± 0.127 | 0.372 ± 0.035 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 10.124 ± 0.516 | 12.458 ± 1.576 | 5.582 ± 0.297 | 6.789 ± 0.097 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 78.209 ± 5.461 | 64.058 ± 7.768 | 46.227 ± 8.496 | 47.877 ± 0.945 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 5.155 ± 0.164 | 12.051 ± 3.535 | 6.289 ± 0.518 | 6.735 ± 0.156 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 39.174 ± 5.077 | 69.384 ± 21.636 | 38.816 ± 5.809 | 45.753 ± 3.697 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.503 ± 0.426 | 2.071 ± 0.228 | 3.131 ± 0.682 | 3.106 ± 0.833 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 29.891 ± 3.410 | 14.259 ± 7.593 | 20.183 ± 5.934 | 20.697 ± 0.986 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.984 ± 0.356 | 2.212 ± 0.736 | 2.024 ± 1.496 | 2.892 ± 0.210 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 22.881 ± 3.819 | 16.384 ± 6.610 | 10.511 ± 2.267 | 19.954 ± 0.760 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.015 ± 0.002 | 0.008 ± 0.002 | 0.011 ± 0.001 | 0.033 ± 0.008 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.062 ± 0.000 | 0.034 ± 0.012 | 0.059 ± 0.002 | 0.077 ± 0.017 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 6.325 ± 0.793 | 6.386 ± 0.941 | 2.934 ± 0.431 | 4.411 ± 0.157 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 32.019 ± 7.400 | 22.939 ± 1.314 | 27.099 ± 4.466 | 35.236 ± 1.651 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 3.378 ± 1.386 | 5.543 ± 1.118 | 3.452 ± 0.163 | 4.620 ± 1.311 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 40.596 ± 9.270 | 27.249 ± 8.068 | 22.620 ± 4.618 | 33.917 ± 4.371 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.073 ± 0.002 | 0.085 ± 0.019 | 0.084 ± 0.010 | 0.053 ± 0.021 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.664 ± 0.075 | 0.855 ± 0.262 | 0.594 ± 0.098 | 0.321 ± 0.066 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 4.172 ± 3.220 | 4.127 ± 4.815 | 3.842 ± 2.272 | 1.798 ± 0.102 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.359 ± 0.039 | 0.931 ± 0.353 | 0.661 ± 0.336 | 0.353 ± 0.071 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 2.744 ± 1.784 | 5.009 ± 1.409 | 3.231 ± 0.115 | 1.568 ± 0.069 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.527 ± 0.021 | 0.696 ± 0.036 | 0.642 ± 0.075 | 0.552 ± 0.032 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 21.541 ± 2.086 | 19.751 ± 4.153 | 13.759 ± 4.683 | 12.098 ± 0.558 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 163.039 ± 6.594 | 135.279 ± 6.563 | 94.224 ± 21.357 | 75.602 ± 5.973 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 10.232 ± 0.471 | 20.122 ± 2.902 | 13.471 ± 3.516 | 11.129 ± 0.196 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 72.936 ± 22.795 | 116.524 ± 22.008 | 76.338 ± 13.087 | 78.520 ± 5.762 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.029 ± 0.006 | 0.025 ± 0.004 | 0.057 ± 0.040 | 0.140 ± 0.018 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.188 ± 0.017 | 0.153 ± 0.046 | 0.334 ± 0.475 | 1.026 ± 0.459 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.582 ± 0.040 | 0.579 ± 0.047 | 0.688 ± 0.108 | 3.613 ± 0.382 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.069 ± 0.005 | 0.093 ± 0.024 | 0.093 ± 0.104 | 0.082 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.023 ± 0.001 | 0.026 ± 0.004 | 0.036 ± 0.017 | 0.044 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.002 | 0.027 ± 0.005 | 0.054 ± 0.011 | 0.088 ± 0.035 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.027 ± 0.000 | 0.034 ± 0.003 | 0.056 ± 0.009 | 0.162 ± 0.036 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.493 ± 0.036 | 0.512 ± 0.052 | 0.454 ± 0.240 | 0.529 ± 0.086 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.004 ± 0.001 | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.011 | 0.009 ± 0.001 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.007 ± 0.001 | 0.007 ± 0.005 | 0.018 ± 0.008 | 0.012 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.003 ± 0.000 | 0.008 ± 0.004 | 0.013 ± 0.001 | 0.067 ± 0.006 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.005 ± 0.002 | 0.012 ± 0.006 | 0.080 ± 0.012 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.006 | 0.094 ± 0.009 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.113 ± 0.008 | 0.016 ± 0.017 | 0.126 ± 0.066 | 0.006 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.116 ± 0.013 | 0.019 ± 0.001 | 0.087 ± 0.005 | 0.014 ± 0.011 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.139 ± 0.019 | 0.028 ± 0.000 | 0.082 ± 0.006 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.037 ± 0.002 | 0.017 ± 0.001 | 0.080 ± 0.077 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.036 ± 0.001 | 0.020 ± 0.002 | 0.078 ± 0.031 | 0.013 ± 0.011 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.052 ± 0.002 | 0.028 ± 0.010 | 0.073 ± 0.002 | 0.011 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.393 ± 0.061 | 0.020 ± 0.008 | 0.111 ± 0.091 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.359 ± 0.010 | 0.021 ± 0.003 | 0.114 ± 0.029 | 0.014 ± 0.008 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.437 ± 0.053 | 0.025 ± 0.019 | 0.127 ± 0.022 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.097 ± 0.026 | 0.019 ± 0.020 | 0.091 ± 0.042 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.082 ± 0.004 | 0.018 ± 0.001 | 0.103 ± 0.021 | 0.012 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.103 ± 0.008 | 0.028 ± 0.008 | 0.099 ± 0.004 | 0.008 ± 0.014 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.039 ± 0.015 | 0.008 ± 0.001 | 0.054 ± 0.029 | 0.006 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.032 ± 0.001 | 0.007 ± 0.000 | 0.037 ± 0.009 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.035 ± 0.005 | 0.010 ± 0.001 | 0.041 ± 0.019 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.598 ± 0.516 | 0.037 ± 0.009 | 0.101 ± 0.025 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.481 ± 0.015 | 0.028 ± 0.018 | 0.085 ± 0.030 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.548 ± 0.030 | 0.039 ± 0.012 | 0.090 ± 0.038 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.081 ± 0.025 | 0.034 ± 0.006 | 0.080 ± 0.004 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.071 ± 0.004 | 0.031 ± 0.002 | 0.101 ± 0.030 | 0.012 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.083 ± 0.006 | 0.051 ± 0.046 | 0.086 ± 0.001 | 0.008 ± 0.004 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.046 ± 0.006 | 0.031 ± 0.010 | 0.042 ± 0.003 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.043 ± 0.002 | 0.021 ± 0.022 | 0.063 ± 0.007 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.050 ± 0.006 | 0.023 ± 0.001 | 0.068 ± 0.044 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.102 ± 0.078 | 0.016 ± 0.008 | 0.167 ± 0.011 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.088 ± 0.026 | 0.016 ± 0.005 | 0.205 ± 0.054 | 0.012 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.092 ± 0.011 | 0.029 ± 0.016 | 0.185 ± 0.026 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.074 ± 0.011 | 0.015 ± 0.002 | 0.083 ± 0.042 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.035 ± 0.002 | 0.015 ± 0.001 | 0.097 ± 0.034 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.043 ± 0.010 | 0.024 ± 0.006 | 0.083 ± 0.016 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.044 ± 0.020 | 0.013 ± 0.000 | 0.044 ± 0.025 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.039 ± 0.004 | 0.020 ± 0.007 | 0.051 ± 0.020 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.048 ± 0.001 | 0.028 ± 0.004 | 0.044 ± 0.018 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.106 ± 0.005 | 0.015 ± 0.005 | 0.107 ± 0.008 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.091 ± 0.010 | 0.021 ± 0.009 | 0.150 ± 0.042 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.116 ± 0.004 | 0.035 ± 0.005 | 0.117 ± 0.006 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.039 ± 0.010 | 0.016 ± 0.007 | 0.084 ± 0.054 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.034 ± 0.001 | 0.015 ± 0.004 | 0.081 ± 0.025 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.046 ± 0.009 | 0.030 ± 0.007 | 0.074 ± 0.018 | 0.020 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.002 ± 0.001 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.003 ± 0.000 | 0.007 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.000 | 0.002 ± 0.000 | 0.001 ± 0.001 | 0.005 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.008 ± 0.001 | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.024 ± 0.037 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.002 ± 0.000 | 0.002 ± 0.001 | 0.001 ± 0.001 | 0.007 ± 0.005 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.004 ± 0.002 | 0.003 ± 0.001 | 0.005 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.001 | 0.004 ± 0.002 | 0.008 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.008 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.002 | 0.012 ± 0.007 | 0.010 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.005 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.003 ± 0.000 | 0.007 ± 0.006 | 0.010 ± 0.013 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.003 ± 0.001 | 0.004 ± 0.001 | 0.011 ± 0.009 | 0.006 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.015 ± 0.011 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.005 | 0.008 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.005 ± 0.002 | 0.005 ± 0.002 | 0.008 ± 0.004 | 0.012 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.002 | 0.015 ± 0.010 | 0.015 ± 0.002 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.014 ± 0.001 | 0.013 ± 0.001 | 0.017 ± 0.003 | 0.022 ± 0.002 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 42.9x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 40.0x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 33.3x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.067 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 36.6x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 45.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 14.0x faster than the slowest successful cell (`tenferro-eager`, 0.139 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 48.4x faster than the slowest successful cell (`tenferro-eager`, 0.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`tenferro-eager`, 0.393 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 25.8x faster than the slowest successful cell (`tenferro-eager`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.359 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 49.3x faster than the slowest successful cell (`tenferro-eager`, 0.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.437 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.0x faster than the slowest successful cell (`tenferro-eager`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 85.4x faster than the slowest successful cell (`tenferro-eager`, 0.598 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`tenferro-eager`, 0.598 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 50.8x faster than the slowest successful cell (`tenferro-eager`, 0.481 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 0.481 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 69.0x faster than the slowest successful cell (`tenferro-eager`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`tenferro-eager`, 0.548 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 22.7x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.167 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.185 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.150 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261115_175655/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261115_175655/run.yaml`
- Timestamp: `2026-11-15T17:56:55.332197+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261115_175655`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

### CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

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

- tenferro-rs features: `blas-accelerate`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

### Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`


- CSV: `data/results/mac-cpu/cpu/einsum/20261115_175655/cpu_ops_t4_20261115_175655.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.008 ± 0.000 | 0.018 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.002 | 0.020 ± 0.000 | 0.020 ± 0.001 | 0.078 ± 0.037 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.025 ± 0.001 | 0.024 ± 0.003 | 0.028 ± 0.006 | 0.033 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.084 ± 0.014 | 0.082 ± 0.002 | 0.090 ± 0.015 | 0.138 ± 0.037 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.004 | 0.080 ± 0.016 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | 0.147 ± 0.066 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.049 ± 0.002 | 0.069 ± 0.034 | 0.114 ± 0.026 | 0.286 ± 0.203 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.004 | 0.112 ± 0.028 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.005 | 0.096 ± 0.020 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.002 | 0.046 ± 0.010 | 0.016 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.017 ± 0.001 | 0.016 ± 0.001 | 0.072 ± 0.028 | 0.034 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.001 | 0.012 ± 0.001 | 0.056 ± 0.006 | 0.017 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.045 ± 0.003 | 0.037 ± 0.002 | 0.103 ± 0.012 | 0.088 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.000 | 0.010 ± 0.004 | 0.045 ± 0.028 | 0.011 ± 0.017 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.013 ± 0.001 | 0.012 ± 0.002 | 0.071 ± 0.014 | 0.035 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.069 ± 0.026 | 0.012 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.017 ± 0.001 | 0.015 ± 0.000 | 0.056 ± 0.041 | 0.053 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.019 ± 0.004 | 0.015 ± 0.000 | 0.056 ± 0.040 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.044 ± 0.002 | 0.042 ± 0.012 | 0.046 ± 0.025 | 0.189 ± 0.057 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.049 ± 0.003 | 0.046 ± 0.001 | 0.057 ± 0.033 | 0.079 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.208 ± 0.029 | 0.162 ± 0.002 | 0.191 ± 0.007 | 0.334 ± 0.029 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.044 ± 0.004 | 0.012 ± 0.000 | 0.058 ± 0.011 | 0.009 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.081 ± 0.056 | 0.016 ± 0.001 | 0.077 ± 0.012 | 0.009 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.003 | 0.013 ± 0.000 | 0.073 ± 0.018 | 0.029 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.053 ± 0.005 | 0.020 ± 0.000 | 0.065 ± 0.003 | 0.052 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.064 ± 0.002 | 0.039 ± 0.002 | 0.102 ± 0.028 | 0.016 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.087 ± 0.099 | 0.057 ± 0.001 | 0.200 ± 0.069 | 0.028 ± 0.021 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.060 ± 0.001 | 0.044 ± 0.002 | 0.117 ± 0.018 | 0.026 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.080 ± 0.041 | 0.070 ± 0.002 | 0.109 ± 0.010 | 0.046 ± 0.027 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.311 ± 0.020 | 0.326 ± 0.003 | 0.310 ± 0.035 | 0.475 ± 0.397 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 9.684 ± 0.710 | 8.675 ± 0.324 | 6.290 ± 0.333 | 7.034 ± 5.794 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 74.469 ± 6.712 | 58.884 ± 3.401 | 41.652 ± 8.425 | 41.021 ± 4.385 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.535 ± 0.295 | 8.684 ± 0.309 | 5.658 ± 0.404 | 6.634 ± 1.903 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 39.127 ± 7.108 | 65.623 ± 2.654 | 36.932 ± 4.754 | 59.147 ± 10.550 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 2.808 ± 0.115 | 1.289 ± 0.086 | 3.720 ± 3.720 | 4.004 ± 1.025 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 27.799 ± 8.368 | 13.567 ± 1.858 | 19.984 ± 6.542 | 24.414 ± 7.687 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.664 ± 0.060 | 1.440 ± 0.089 | 4.715 ± 0.899 | 2.517 ± 0.133 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 16.963 ± 1.511 | 13.229 ± 1.299 | 11.236 ± 0.946 | 24.330 ± 6.137 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.009 ± 0.000 | 0.011 ± 0.004 | 0.042 ± 0.009 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.066 ± 0.002 | 0.031 ± 0.003 | 0.051 ± 0.005 | 0.118 ± 0.028 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 5.976 ± 0.228 | 4.057 ± 0.161 | 3.890 ± 0.298 | 6.077 ± 4.373 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 55.064 ± 5.517 | 27.283 ± 4.716 | 16.327 ± 0.508 | 29.585 ± 4.978 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.844 ± 0.235 | 3.999 ± 0.116 | 3.596 ± 0.261 | 4.259 ± 0.551 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 36.501 ± 3.824 | 26.671 ± 1.895 | 33.256 ± 9.861 | 29.758 ± 14.661 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.092 ± 0.026 | 0.102 ± 0.004 | 0.088 ± 0.012 | 0.084 ± 0.028 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.806 ± 0.088 | 0.566 ± 0.010 | 0.560 ± 0.022 | 0.387 ± 0.021 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 3.416 ± 0.537 | 3.995 ± 0.157 | 2.537 ± 0.622 | 2.213 ± 0.149 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.451 ± 0.043 | 0.543 ± 0.011 | 0.732 ± 0.023 | 0.429 ± 0.021 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.790 ± 0.124 | 3.809 ± 0.923 | 3.731 ± 0.438 | 2.030 ± 0.742 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.504 ± 0.080 | 0.627 ± 0.023 | 0.619 ± 0.059 | 0.654 ± 0.255 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 19.580 ± 0.423 | 15.215 ± 0.766 | 12.385 ± 2.438 | 11.268 ± 1.439 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 137.659 ± 10.773 | 87.715 ± 2.140 | 65.683 ± 12.400 | 70.142 ± 8.747 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 9.856 ± 0.107 | 15.486 ± 1.201 | 10.711 ± 1.192 | 10.507 ± 0.018 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 79.588 ± 13.876 | 90.287 ± 1.040 | 54.545 ± 3.669 | 67.438 ± 1.796 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.030 ± 0.001 | 0.023 ± 0.004 | 0.027 ± 0.002 | 0.226 ± 0.126 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.182 ± 0.054 | 0.134 ± 0.009 | 0.184 ± 0.017 | 0.678 ± 0.189 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 0.495 ± 0.027 | 0.550 ± 0.061 | 0.717 ± 0.170 | 2.320 ± 1.441 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.065 ± 0.009 | 0.070 ± 0.004 | 0.124 ± 0.017 | 0.108 ± 0.100 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.022 ± 0.000 | 0.025 ± 0.002 | 0.032 ± 0.009 | 0.092 ± 0.038 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.004 | 0.028 ± 0.001 | 0.046 ± 0.010 | 0.119 ± 0.057 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.028 ± 0.000 | 0.034 ± 0.000 | 0.066 ± 0.008 | 0.256 ± 0.055 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.444 ± 0.014 | 0.479 ± 0.027 | 0.456 ± 0.015 | 0.541 ± 0.046 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.009 ± 0.003 | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.008 ± 0.004 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.005 ± 0.002 | 0.010 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.012 ± 0.002 | 0.008 ± 0.001 | 0.010 ± 0.003 | 0.014 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.012 ± 0.004 | 0.082 ± 0.125 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.003 ± 0.001 | 0.006 ± 0.000 | 0.009 ± 0.003 | 0.084 ± 0.016 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.007 ± 0.000 | 0.012 ± 0.004 | 0.090 ± 0.009 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.118 ± 0.016 | 0.023 ± 0.000 | 0.083 ± 0.007 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.136 ± 0.015 | 0.025 ± 0.002 | 0.070 ± 0.002 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.150 ± 0.010 | 0.040 ± 0.002 | 0.114 ± 0.028 | 0.026 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.042 ± 0.013 | 0.023 ± 0.003 | 0.073 ± 0.006 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.049 ± 0.001 | 0.025 ± 0.000 | 0.060 ± 0.001 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.059 ± 0.003 | 0.036 ± 0.003 | 0.101 ± 0.036 | 0.026 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.367 ± 0.022 | 0.027 ± 0.002 | 0.425 ± 0.231 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.441 ± 0.022 | 0.035 ± 0.018 | 0.399 ± 0.224 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.374 ± 0.024 | 0.035 ± 0.004 | 0.428 ± 0.069 | 0.014 ± 0.003 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.110 ± 0.079 | 0.026 ± 0.000 | 0.341 ± 0.037 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.147 ± 0.017 | 0.025 ± 0.002 | 0.179 ± 0.054 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.129 ± 0.007 | 0.051 ± 0.009 | 0.322 ± 0.061 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.127 ± 0.015 | 0.011 ± 0.002 | 0.029 ± 0.003 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.045 ± 0.002 | 0.009 ± 0.000 | 0.025 ± 0.001 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.052 ± 0.014 | 0.013 ± 0.001 | 0.040 ± 0.026 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.587 ± 0.055 | 0.040 ± 0.001 | 0.201 ± 0.044 | 0.009 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.582 ± 0.046 | 0.040 ± 0.002 | 0.285 ± 0.132 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.650 ± 0.038 | 0.106 ± 0.028 | 0.234 ± 0.081 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.127 ± 0.035 | 0.042 ± 0.003 | 0.172 ± 0.023 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.100 ± 0.018 | 0.039 ± 0.001 | 0.236 ± 0.065 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.134 ± 0.018 | 0.053 ± 0.013 | 0.188 ± 0.099 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.065 ± 0.122 | 0.031 ± 0.001 | 0.045 ± 0.003 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.061 ± 0.004 | 0.033 ± 0.002 | 0.041 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.053 ± 0.003 | 0.047 ± 0.008 | 0.090 ± 0.088 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.083 ± 0.023 | 0.021 ± 0.000 | 0.180 ± 0.018 | 0.009 ± 0.002 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.097 ± 0.011 | 0.021 ± 0.001 | 0.161 ± 0.011 | 0.007 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.084 ± 0.005 | 0.026 ± 0.018 | 0.226 ± 0.114 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.046 ± 0.002 | 0.022 ± 0.003 | 0.087 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.052 ± 0.004 | 0.032 ± 0.004 | 0.079 ± 0.010 | 0.008 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.043 ± 0.001 | 0.026 ± 0.001 | 0.118 ± 0.032 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.058 ± 0.006 | 0.017 ± 0.001 | 0.036 ± 0.008 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.050 ± 0.006 | 0.020 ± 0.001 | 0.033 ± 0.001 | 0.014 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.065 ± 0.126 | 0.040 ± 0.003 | 0.095 ± 0.049 | 0.039 ± 0.016 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.119 ± 0.042 | 0.017 ± 0.001 | 0.115 ± 0.027 | 0.014 ± 0.014 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.118 ± 0.003 | 0.020 ± 0.001 | 0.148 ± 0.051 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.162 ± 0.017 | 0.047 ± 0.005 | 0.142 ± 0.032 | 0.042 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.052 ± 0.022 | 0.016 ± 0.001 | 0.085 ± 0.024 | 0.012 ± 0.004 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.070 ± 0.014 | 0.018 ± 0.000 | 0.097 ± 0.047 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.050 ± 0.006 | 0.039 ± 0.002 | 0.085 ± 0.010 | 0.043 ± 0.037 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.002 | 0.004 ± 0.000 | 0.001 ± 0.001 | 0.009 ± 0.005 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.001 | 0.005 ± 0.002 | 0.002 ± 0.001 | 0.017 ± 0.010 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.009 ± 0.002 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.075 ± 0.015 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.007 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.004 ± 0.001 | 0.027 ± 0.004 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.005 ± 0.001 | 0.004 ± 0.000 | 0.044 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.030 ± 0.009 | 0.010 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.011 ± 0.004 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.006 ± 0.002 | 0.006 ± 0.001 | 0.010 ± 0.000 | 0.007 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.008 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.011 ± 0.008 | 0.007 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.005 | 0.008 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.011 ± 0.004 | 0.005 ± 0.000 | 0.008 ± 0.002 | 0.013 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.009 ± 0.001 | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.002 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.017 ± 0.005 | 0.014 ± 0.001 | 0.022 ± 0.015 | 0.020 ± 0.001 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 25.6x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `pytorch-cpu` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 29.6x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 28.3x faster than the slowest successful cell (`jax-cpu`, 0.147 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.112 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 30.9x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 29.1x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 15.2x faster than the slowest successful cell (`tenferro-eager`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 49.5x faster than the slowest successful cell (`pytorch-cpu`, 0.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 0.425 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 50.4x faster than the slowest successful cell (`tenferro-eager`, 0.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.5x faster than the slowest successful cell (`tenferro-eager`, 0.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 31.0x faster than the slowest successful cell (`pytorch-cpu`, 0.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 42.9x faster than the slowest successful cell (`pytorch-cpu`, 0.341 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.341 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.2x faster than the slowest successful cell (`tenferro-eager`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.127 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 63.8x faster than the slowest successful cell (`tenferro-eager`, 0.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 59.3x faster than the slowest successful cell (`tenferro-eager`, 0.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`tenferro-eager`, 0.582 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 42.3x faster than the slowest successful cell (`tenferro-eager`, 0.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.118 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`64x64`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/qr` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

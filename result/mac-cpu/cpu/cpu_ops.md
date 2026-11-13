# CPU Benchmark Results

- Target profile: `mac-cpu`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Benchmark commit: `f3a05ec9a2a7262482348fd5bddccbe90389866b`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/mac-cpu/cpu/einsum/20261113_172248/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_172248/run.yaml`
- Timestamp: `2026-11-13T17:22:48.940279+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261113_172248`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261113_172248/cpu_ops_t1_20261113_172248.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.001 | 0.008 ± 0.001 | 0.008 ± 0.001 | 0.018 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.025 ± 0.001 | 0.023 ± 0.001 | 0.032 ± 0.021 | 0.046 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.029 ± 0.001 | 0.029 ± 0.003 | 0.029 ± 0.002 | 0.042 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.097 ± 0.003 | 0.101 ± 0.009 | 0.106 ± 0.015 | 0.165 ± 0.117 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.004 ± 0.001 | 0.005 ± 0.001 | 0.014 ± 0.011 | 0.065 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.094 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.061 ± 0.003 | 0.084 ± 0.022 | 0.071 ± 0.064 | 0.118 ± 0.127 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.003 | 0.210 ± 0.130 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.012 | 0.132 ± 0.054 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.002 | 0.013 ± 0.003 | 0.013 ± 0.006 | 0.014 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.020 ± 0.001 | 0.021 ± 0.002 | 0.035 ± 0.007 | 0.039 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.019 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.044 ± 0.002 | 0.048 ± 0.005 | 0.044 ± 0.007 | 0.057 ± 0.023 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.012 ± 0.002 | 0.011 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.014 ± 0.000 | 0.013 ± 0.001 | 0.024 ± 0.011 | 0.021 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.003 | 0.012 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.020 ± 0.001 | 0.018 ± 0.002 | 0.026 ± 0.003 | 0.028 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.018 ± 0.006 | 0.017 ± 0.000 | 0.021 ± 0.002 | 0.042 ± 0.019 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.054 ± 0.010 | 0.051 ± 0.002 | 0.057 ± 0.002 | 0.159 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.055 ± 0.002 | 0.054 ± 0.001 | 0.057 ± 0.018 | 0.091 ± 0.121 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.212 ± 0.010 | 0.200 ± 0.028 | 0.164 ± 0.031 | 0.608 ± 0.165 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.036 ± 0.005 | 0.010 ± 0.000 | 0.073 ± 0.006 | 0.012 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.050 ± 0.040 | 0.017 ± 0.001 | 0.152 ± 0.056 | 0.009 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.078 ± 0.054 | 0.012 ± 0.000 | 0.067 ± 0.005 | 0.026 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.054 ± 0.004 | 0.023 ± 0.005 | 0.103 ± 0.022 | 0.036 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.055 ± 0.005 | 0.035 ± 0.001 | 0.056 ± 0.014 | 0.018 ± 0.030 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.065 ± 0.001 | 0.062 ± 0.006 | 0.086 ± 0.049 | 0.024 ± 0.010 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.061 ± 0.053 | 0.041 ± 0.003 | 0.074 ± 0.033 | 0.022 ± 0.013 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.078 ± 0.006 | 0.080 ± 0.066 | 0.102 ± 0.023 | 0.034 ± 0.013 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.347 ± 0.036 | 0.349 ± 0.017 | 0.418 ± 0.191 | 0.413 ± 0.071 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 11.869 ± 0.248 | 9.612 ± 0.471 | 7.023 ± 2.268 | 8.637 ± 2.028 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 68.239 ± 10.794 | 55.743 ± 3.309 | 45.577 ± 13.738 | 62.110 ± 5.214 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 6.140 ± 1.286 | 9.372 ± 0.753 | 5.700 ± 0.521 | 8.784 ± 4.492 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 36.506 ± 4.852 | 53.868 ± 2.455 | 33.949 ± 22.827 | 47.086 ± 10.271 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 3.313 ± 0.608 | 1.354 ± 0.035 | 2.558 ± 0.207 | 2.928 ± 0.265 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 19.388 ± 0.887 | 12.347 ± 2.302 | 19.719 ± 11.764 | 26.997 ± 5.904 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 2.625 ± 0.252 | 1.508 ± 0.104 | 1.594 ± 0.425 | 5.687 ± 2.054 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 19.898 ± 1.267 | 12.539 ± 4.042 | 11.115 ± 7.215 | 31.519 ± 6.650 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.018 ± 0.003 | 0.008 ± 0.000 | 0.014 ± 0.022 | 0.043 ± 0.021 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.067 ± 0.004 | 0.028 ± 0.001 | 0.061 ± 0.075 | 0.070 ± 0.008 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 6.557 ± 0.218 | 4.459 ± 0.293 | 3.553 ± 3.009 | 9.088 ± 5.942 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 37.242 ± 8.269 | 22.373 ± 0.970 | 24.335 ± 3.399 | 36.262 ± 3.421 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 4.170 ± 0.217 | 4.553 ± 0.153 | 3.221 ± 0.400 | 10.168 ± 2.704 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 33.225 ± 16.011 | 22.389 ± 3.767 | 23.683 ± 3.343 | 36.357 ± 7.240 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.079 ± 0.011 | 0.086 ± 0.005 | 0.086 ± 0.032 | 0.043 ± 0.016 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.839 ± 0.104 | 0.632 ± 0.030 | 0.550 ± 0.031 | 0.597 ± 0.273 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 4.200 ± 0.614 | 3.355 ± 0.801 | 2.713 ± 3.017 | 2.164 ± 0.271 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.513 ± 0.110 | 0.549 ± 0.025 | 0.713 ± 0.115 | 0.777 ± 0.223 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 2.138 ± 0.242 | 3.473 ± 0.740 | 4.841 ± 2.583 | 1.800 ± 0.187 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.695 ± 0.444 | 0.638 ± 0.072 | 0.559 ± 0.060 | 0.518 ± 0.053 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 21.404 ± 1.237 | 16.743 ± 1.241 | 16.188 ± 7.072 | 13.421 ± 1.348 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 127.501 ± 12.786 | 104.009 ± 4.205 | 79.142 ± 15.616 | 81.714 ± 17.910 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 10.720 ± 0.553 | 16.366 ± 1.742 | 11.167 ± 0.405 | 13.845 ± 4.125 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 60.245 ± 1.601 | 104.808 ± 7.185 | 67.967 ± 7.273 | 74.641 ± 24.574 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.053 ± 0.050 | 0.022 ± 0.004 | 0.033 ± 0.007 | 0.167 ± 0.075 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 0.338 ± 0.162 | 0.153 ± 0.127 | 0.245 ± 0.211 | 1.216 ± 0.443 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 0.607 ± 0.192 | 0.558 ± 0.174 | 1.215 ± 1.255 | 4.084 ± 1.248 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.077 ± 0.002 | 0.073 ± 0.002 | 0.120 ± 0.070 | 0.136 ± 0.030 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.022 ± 0.005 | 0.022 ± 0.001 | 0.033 ± 0.004 | 0.066 ± 0.040 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.022 ± 0.001 | 0.025 ± 0.006 | 0.061 ± 0.060 | 0.113 ± 0.070 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.028 ± 0.000 | 0.032 ± 0.002 | 0.132 ± 0.120 | 0.238 ± 0.184 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.565 ± 0.072 | 0.497 ± 0.022 | 0.574 ± 0.122 | 0.977 ± 0.333 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.007 ± 0.004 | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.009 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.004 ± 0.003 | 0.004 ± 0.002 | 0.006 ± 0.001 | 0.011 ± 0.006 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.010 ± 0.004 | 0.011 ± 0.007 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.136 ± 0.086 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.004 ± 0.003 | 0.006 ± 0.004 | 0.011 ± 0.001 | 0.146 ± 0.082 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.003 | 0.006 ± 0.003 | 0.012 ± 0.001 | 0.169 ± 0.047 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.119 ± 0.023 | 0.025 ± 0.008 | 0.103 ± 0.121 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.245 ± 0.101 | 0.026 ± 0.023 | 0.084 ± 0.003 | 0.016 ± 0.011 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.156 ± 0.006 | 0.027 ± 0.001 | 0.240 ± 0.195 | 0.022 ± 0.015 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.049 ± 0.059 | 0.023 ± 0.008 | 0.091 ± 0.022 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.122 ± 0.022 | 0.033 ± 0.008 | 0.071 ± 0.003 | 0.022 ± 0.004 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.050 ± 0.004 | 0.027 ± 0.001 | 0.125 ± 0.122 | 0.010 ± 0.016 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.623 ± 0.306 | 0.026 ± 0.008 | 0.155 ± 0.013 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.649 ± 0.661 | 0.022 ± 0.007 | 0.125 ± 0.007 | 0.015 ± 0.004 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.477 ± 0.087 | 0.022 ± 0.002 | 0.212 ± 0.032 | 0.017 ± 0.009 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.095 ± 0.089 | 0.029 ± 0.016 | 0.119 ± 0.051 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.143 ± 0.090 | 0.022 ± 0.017 | 0.096 ± 0.002 | 0.016 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.169 ± 0.098 | 0.024 ± 0.002 | 0.143 ± 0.085 | 0.018 ± 0.006 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.117 ± 0.060 | 0.008 ± 0.000 | 0.037 ± 0.023 | 0.012 ± 0.014 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.032 ± 0.001 | 0.020 ± 0.003 | 0.030 ± 0.004 | 0.015 ± 0.008 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.040 ± 0.003 | 0.009 ± 0.000 | 0.046 ± 0.009 | 0.011 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.883 ± 0.349 | 0.054 ± 0.012 | 0.111 ± 0.013 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 1.402 ± 0.501 | 0.039 ± 0.009 | 0.104 ± 0.026 | 0.011 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.604 ± 0.125 | 0.034 ± 0.001 | 0.100 ± 0.035 | 0.018 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.172 ± 0.169 | 0.036 ± 0.018 | 0.118 ± 0.011 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.209 ± 0.033 | 0.048 ± 0.061 | 0.091 ± 0.004 | 0.014 ± 0.006 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.087 ± 0.005 | 0.036 ± 0.001 | 0.257 ± 0.174 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.061 ± 0.205 | 0.024 ± 0.001 | 0.053 ± 0.002 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.126 ± 0.007 | 0.028 ± 0.006 | 0.048 ± 0.003 | 0.015 ± 0.004 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.055 ± 0.002 | 0.024 ± 0.000 | 0.070 ± 0.037 | 0.018 ± 0.006 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.134 ± 0.021 | 0.017 ± 0.005 | 0.282 ± 0.159 | 0.011 ± 0.003 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.193 ± 0.057 | 0.029 ± 0.009 | 0.181 ± 0.010 | 0.017 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.131 ± 0.072 | 0.020 ± 0.001 | 0.334 ± 0.264 | 0.018 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.053 ± 0.027 | 0.016 ± 0.006 | 0.092 ± 0.023 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.132 ± 0.160 | 0.018 ± 0.004 | 0.092 ± 0.003 | 0.013 ± 0.011 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.058 ± 0.046 | 0.018 ± 0.001 | 0.282 ± 0.211 | 0.016 ± 0.008 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.047 ± 0.003 | 0.013 ± 0.002 | 0.041 ± 0.002 | 0.013 ± 0.014 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.075 ± 0.060 | 0.027 ± 0.028 | 0.039 ± 0.006 | 0.035 ± 0.031 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.055 ± 0.003 | 0.028 ± 0.011 | 0.085 ± 0.056 | 0.024 ± 0.003 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.266 ± 0.145 | 0.017 ± 0.020 | 0.144 ± 0.033 | 0.011 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.342 ± 0.117 | 0.043 ± 0.014 | 0.119 ± 0.011 | 0.012 ± 0.005 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.143 ± 0.008 | 0.034 ± 0.020 | 0.206 ± 0.097 | 0.020 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.069 ± 0.071 | 0.013 ± 0.002 | 0.088 ± 0.017 | 0.011 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.088 ± 0.033 | 0.034 ± 0.024 | 0.076 ± 0.002 | 0.014 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.058 ± 0.013 | 0.029 ± 0.000 | 0.125 ± 0.131 | 0.026 ± 0.009 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.002 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.006 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.004 ± 0.001 | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.014 ± 0.011 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.002 ± 0.003 | 0.004 ± 0.001 | 0.001 ± 0.000 | 0.008 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.027 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.003 ± 0.002 | 0.003 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.009 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.008 ± 0.002 | 0.005 ± 0.002 | 0.004 ± 0.001 | 0.009 ± 0.002 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.009 ± 0.006 | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.012 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.007 ± 0.004 | 0.005 ± 0.001 | 0.011 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.008 ± 0.003 | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.008 ± 0.002 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.004 ± 0.004 | 0.006 ± 0.001 | 0.010 ± 0.001 | 0.013 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.004 ± 0.002 | 0.006 ± 0.002 | 0.011 ± 0.011 | 0.007 ± 0.005 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.018 ± 0.008 | 0.011 ± 0.006 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.019 ± 0.013 | 0.008 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.004 | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.010 ± 0.003 | 0.008 ± 0.001 | 0.010 ± 0.001 | 0.017 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.035 ± 0.010 | 0.017 ± 0.004 | 0.018 ± 0.001 | 0.031 ± 0.015 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 57.7x faster than the slowest successful cell (`jax-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 59.6x faster than the slowest successful cell (`jax-cpu`, 0.210 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `pytorch-cpu` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 55.3x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 36.4x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 36.1x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.7x faster than the slowest successful cell (`tenferro-eager`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.240 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 54.2x faster than the slowest successful cell (`tenferro-eager`, 0.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 0.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 42.6x faster than the slowest successful cell (`tenferro-eager`, 0.649 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 29.9x faster than the slowest successful cell (`tenferro-eager`, 0.649 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 28.2x faster than the slowest successful cell (`tenferro-eager`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.117 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 90.4x faster than the slowest successful cell (`tenferro-eager`, 0.883 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`tenferro-eager`, 0.883 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 133.0x faster than the slowest successful cell (`tenferro-eager`, 1.402 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 1.402 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 35.8x faster than the slowest successful cell (`tenferro-eager`, 1.402 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 33.2x faster than the slowest successful cell (`tenferro-eager`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`tenferro-eager`, 0.604 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 16.8x faster than the slowest successful cell (`tenferro-eager`, 0.172 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 15.3x faster than the slowest successful cell (`tenferro-eager`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 18.0x faster than the slowest successful cell (`pytorch-cpu`, 0.257 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.193 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 18.1x faster than the slowest successful cell (`pytorch-cpu`, 0.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 0.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 24.6x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 27.5x faster than the slowest successful cell (`tenferro-eager`, 0.342 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.206 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/mac-cpu/cpu/einsum/20261113_180121/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/mac-cpu/cpu/einsum/20261113_180121/run.yaml`
- Timestamp: `2026-11-13T18:01:21.706892+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/mac-cpu/cpu/einsum/20261113_180121`.

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


- CSV: `data/results/mac-cpu/cpu/einsum/20261113_180121/cpu_ops_t4_20261113_180121.csv`

### CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.002 | 0.009 ± 0.001 | 0.010 ± 0.002 | 0.021 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.021 ± 0.001 | 0.026 ± 0.002 | 0.024 ± 0.016 | 0.075 ± 0.014 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.027 ± 0.001 | 0.026 ± 0.001 | 0.030 ± 0.004 | 0.036 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.092 ± 0.003 | 0.090 ± 0.003 | 0.096 ± 0.007 | 0.133 ± 0.019 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.002 | 0.004 ± 0.000 | 0.017 ± 0.004 | 0.125 ± 0.043 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.110 ± 0.067 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.078 ± 0.008 | 0.061 ± 0.001 | 0.114 ± 0.042 | 0.302 ± 0.137 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.004 | 0.196 ± 0.153 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.002 | 0.096 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.046 ± 0.011 | 0.014 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.023 ± 0.001 | 0.066 ± 0.007 | 0.034 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.012 ± 0.001 | 0.012 ± 0.000 | 0.047 ± 0.024 | 0.024 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.042 ± 0.009 | 0.039 ± 0.000 | 0.096 ± 0.025 | 0.093 ± 0.045 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.006 ± 0.001 | 0.009 ± 0.000 | 0.046 ± 0.026 | 0.011 ± 0.006 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.079 ± 0.026 | 0.030 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.008 ± 0.001 | 0.010 ± 0.000 | 0.068 ± 0.017 | 0.012 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.018 ± 0.002 | 0.018 ± 0.006 | 0.057 ± 0.009 | 0.034 ± 0.029 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.020 ± 0.002 | 0.017 ± 0.001 | 0.022 ± 0.007 | 0.055 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.000 | 0.044 ± 0.002 | 0.052 ± 0.003 | 0.167 ± 0.018 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.053 ± 0.003 | 0.057 ± 0.013 | 0.055 ± 0.009 | 0.118 ± 0.051 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.190 ± 0.012 | 0.183 ± 0.004 | 0.204 ± 0.014 | 0.329 ± 0.061 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.043 ± 0.007 | 0.013 ± 0.001 | 0.084 ± 0.038 | 0.009 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.047 ± 0.008 | 0.018 ± 0.000 | 0.066 ± 0.005 | 0.014 ± 0.016 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.052 ± 0.019 | 0.013 ± 0.000 | 0.129 ± 0.064 | 0.031 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.113 ± 0.024 | 0.022 ± 0.001 | 0.073 ± 0.010 | 0.090 ± 0.032 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.069 ± 0.003 | 0.047 ± 0.016 | 0.178 ± 0.027 | 0.015 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.063 ± 0.001 | 0.067 ± 0.001 | 0.123 ± 0.188 | 0.035 ± 0.019 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.081 ± 0.008 | 0.048 ± 0.001 | 0.155 ± 0.049 | 0.011 ± 0.008 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.073 ± 0.007 | 0.082 ± 0.010 | 0.257 ± 0.088 | 0.038 ± 0.043 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.353 ± 0.025 | 0.474 ± 0.101 | 0.343 ± 0.028 | 0.557 ± 0.173 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 11.315 ± 0.817 | 10.003 ± 4.599 | 6.473 ± 0.179 | 5.205 ± 0.092 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 58.316 ± 3.012 | 58.305 ± 1.377 | 33.313 ± 0.188 | 50.020 ± 10.105 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 5.411 ± 2.341 | 8.787 ± 4.692 | 6.184 ± 0.493 | 7.240 ± 2.205 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 40.545 ± 3.743 | 56.548 ± 2.097 | 31.074 ± 4.944 | 34.829 ± 4.918 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 3.591 ± 1.250 | 1.271 ± 0.386 | 3.584 ± 0.122 | 2.613 ± 0.115 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 24.255 ± 3.371 | 10.369 ± 0.521 | 27.721 ± 10.168 | 19.587 ± 11.729 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 2.794 ± 0.387 | 1.508 ± 1.198 | 2.609 ± 0.579 | 2.597 ± 0.478 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 22.547 ± 4.765 | 11.189 ± 0.845 | 16.518 ± 5.660 | 17.275 ± 4.456 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.017 ± 0.000 | 0.009 ± 0.001 | 0.012 ± 0.006 | 0.049 ± 0.007 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.075 ± 0.003 | 0.029 ± 0.000 | 0.048 ± 0.009 | 0.104 ± 0.044 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 5.992 ± 1.013 | 4.140 ± 1.006 | 3.960 ± 0.240 | 5.336 ± 1.054 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 38.110 ± 1.721 | 27.264 ± 1.613 | 29.513 ± 7.727 | 33.605 ± 4.394 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 3.839 ± 0.296 | 4.657 ± 1.646 | 3.681 ± 0.144 | 2.738 ± 0.140 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 26.522 ± 10.687 | 26.911 ± 12.989 | 27.706 ± 1.491 | 38.773 ± 5.289 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.092 ± 0.007 | 0.101 ± 0.006 | 0.085 ± 0.017 | 0.070 ± 0.035 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.899 ± 0.178 | 0.678 ± 0.439 | 0.710 ± 0.084 | 0.471 ± 0.179 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.531 ± 0.100 | 3.895 ± 0.346 | 2.533 ± 0.576 | 2.655 ± 3.040 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.473 ± 0.063 | 0.546 ± 0.022 | 0.904 ± 0.110 | 1.133 ± 0.242 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 3.145 ± 0.806 | 3.564 ± 0.190 | 5.283 ± 2.639 | 1.498 ± 0.098 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.563 ± 0.017 | 0.613 ± 0.008 | 0.488 ± 0.010 | 0.587 ± 0.067 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 23.579 ± 3.526 | 15.007 ± 0.160 | 11.421 ± 1.631 | 13.125 ± 1.291 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 116.721 ± 7.576 | 99.603 ± 6.758 | 72.832 ± 11.886 | 94.419 ± 24.457 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 10.070 ± 3.030 | 19.709 ± 3.009 | 10.892 ± 0.505 | 12.406 ± 1.071 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 65.656 ± 2.652 | 104.267 ± 3.370 | 65.266 ± 4.776 | 79.911 ± 19.847 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.038 ± 0.013 | 0.024 ± 0.006 | 0.027 ± 0.004 | 0.176 ± 0.010 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.204 ± 0.032 | 0.175 ± 0.029 | 0.182 ± 0.016 | 0.575 ± 0.308 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.011 ± 0.766 | 1.096 ± 0.391 | 0.863 ± 0.898 | 2.540 ± 1.345 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.085 ± 0.027 | 0.135 ± 0.037 | 0.143 ± 0.040 | 0.173 ± 0.085 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.023 ± 0.003 | 0.025 ± 0.002 | 0.037 ± 0.004 | 0.066 ± 0.022 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.023 ± 0.002 | 0.029 ± 0.000 | 0.042 ± 0.010 | 0.124 ± 0.024 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.029 ± 0.004 | 0.035 ± 0.003 | 0.069 ± 0.010 | 0.265 ± 0.149 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.497 ± 0.060 | 0.662 ± 0.183 | 0.488 ± 0.024 | 0.503 ± 0.397 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.002 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.008 ± 0.002 | 0.006 ± 0.002 | 0.006 ± 0.001 | 0.017 ± 0.009 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.009 ± 0.001 | 0.010 ± 0.002 | 0.013 ± 0.009 | 0.029 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.009 ± 0.001 | 0.073 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.010 ± 0.002 | 0.084 ± 0.015 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.003 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.000 | 0.245 ± 0.082 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.170 ± 0.037 | 0.026 ± 0.004 | 0.170 ± 0.064 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.180 ± 0.016 | 0.046 ± 0.002 | 0.092 ± 0.006 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.150 ± 0.049 | 0.041 ± 0.021 | 0.095 ± 0.006 | 0.015 ± 0.007 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.059 ± 0.009 | 0.023 ± 0.006 | 0.182 ± 0.118 | 0.009 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.065 ± 0.041 | 0.042 ± 0.010 | 0.077 ± 0.003 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.067 ± 0.002 | 0.045 ± 0.044 | 0.081 ± 0.006 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.607 ± 0.364 | 0.046 ± 0.016 | 0.509 ± 0.441 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.429 ± 0.012 | 0.044 ± 0.013 | 0.477 ± 0.058 | 0.009 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.498 ± 0.011 | 0.051 ± 0.013 | 0.404 ± 0.262 | 0.018 ± 0.005 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.122 ± 0.005 | 0.029 ± 0.002 | 0.232 ± 0.070 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.164 ± 0.038 | 0.036 ± 0.031 | 0.392 ± 0.089 | 0.009 ± 0.004 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.166 ± 0.065 | 0.051 ± 0.021 | 0.845 ± 0.111 | 0.017 ± 0.010 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.040 ± 0.003 | 0.011 ± 0.000 | 0.037 ± 0.015 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.045 ± 0.003 | 0.016 ± 0.004 | 0.030 ± 0.004 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.050 ± 0.004 | 0.013 ± 0.001 | 0.033 ± 0.007 | 0.007 ± 0.003 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 1.120 ± 0.791 | 0.051 ± 0.003 | 0.219 ± 0.196 | 0.016 ± 0.006 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.797 ± 0.219 | 0.062 ± 0.006 | 0.190 ± 0.019 | 0.010 ± 0.002 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.697 ± 0.124 | 0.077 ± 0.015 | 0.194 ± 0.027 | 0.014 ± 0.007 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.181 ± 0.064 | 0.047 ± 0.001 | 0.300 ± 0.041 | 0.009 ± 0.003 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.149 ± 0.074 | 0.075 ± 0.015 | 0.168 ± 0.050 | 0.010 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.126 ± 0.007 | 0.069 ± 0.015 | 0.194 ± 0.046 | 0.015 ± 0.005 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.063 ± 0.017 | 0.038 ± 0.003 | 0.050 ± 0.023 | 0.012 ± 0.003 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.061 ± 0.004 | 0.050 ± 0.014 | 0.053 ± 0.004 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.068 ± 0.047 | 0.056 ± 0.026 | 0.052 ± 0.001 | 0.015 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.120 ± 0.011 | 0.022 ± 0.001 | 0.184 ± 0.016 | 0.007 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.115 ± 0.008 | 0.035 ± 0.015 | 0.203 ± 0.017 | 0.008 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.144 ± 0.042 | 0.047 ± 0.007 | 0.985 ± 0.611 | 0.015 ± 0.006 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.078 ± 0.024 | 0.024 ± 0.001 | 0.087 ± 0.003 | 0.008 ± 0.002 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.069 ± 0.051 | 0.032 ± 0.011 | 0.106 ± 0.016 | 0.025 ± 0.026 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.064 ± 0.069 | 0.034 ± 0.008 | 0.370 ± 0.163 | 0.019 ± 0.010 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.047 ± 0.009 | 0.021 ± 0.002 | 0.059 ± 0.026 | 0.010 ± 0.004 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.058 ± 0.004 | 0.025 ± 0.024 | 0.042 ± 0.007 | 0.020 ± 0.006 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.072 ± 0.096 | 0.038 ± 0.009 | 0.056 ± 0.004 | 0.020 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.106 ± 0.023 | 0.019 ± 0.004 | 0.110 ± 0.038 | 0.012 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.131 ± 0.024 | 0.032 ± 0.027 | 0.128 ± 0.023 | 0.016 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.187 ± 0.054 | 0.051 ± 0.021 | 0.133 ± 0.021 | 0.035 ± 0.019 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.064 ± 0.036 | 0.018 ± 0.000 | 0.120 ± 0.190 | 0.024 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.063 ± 0.022 | 0.021 ± 0.006 | 0.081 ± 0.025 | 0.015 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.054 ± 0.003 | 0.081 ± 0.014 | 0.091 ± 0.004 | 0.045 ± 0.016 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.001 ± 0.000 | 0.005 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.005 ± 0.001 | 0.004 ± 0.002 | 0.002 ± 0.000 | 0.013 ± 0.011 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.001 | 0.005 ± 0.002 | 0.001 ± 0.000 | 0.006 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.009 ± 0.001 | 0.006 ± 0.000 | 0.010 ± 0.014 | 0.031 ± 0.013 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.003 ± 0.001 | 0.004 ± 0.000 | 0.001 ± 0.000 | 0.010 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.007 ± 0.002 | 0.004 ± 0.000 | 0.030 ± 0.015 | 0.010 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.006 ± 0.004 | 0.005 ± 0.001 | 0.048 ± 0.006 | 0.011 ± 0.002 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.039 ± 0.013 | 0.011 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.006 ± 0.009 | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.006 ± 0.002 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.005 ± 0.003 | 0.008 ± 0.001 | 0.012 ± 0.006 | 0.006 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.009 ± 0.001 | 0.011 ± 0.002 | 0.010 ± 0.002 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.006 ± 0.003 | 0.010 ± 0.005 | 0.011 ± 0.001 | 0.006 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.004 | 0.017 ± 0.008 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.006 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.006 ± 0.001 | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.012 ± 0.004 | 0.010 ± 0.001 | 0.009 ± 0.001 | 0.021 ± 0.018 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.015 ± 0.005 | 0.019 ± 0.010 | 0.018 ± 0.010 | 0.038 ± 0.014 | - | - |

### Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 18.7x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 29.4x faster than the slowest successful cell (`jax-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 23.0x faster than the slowest successful cell (`jax-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`jax-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 53.6x faster than the slowest successful cell (`jax-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`jax-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`2x2xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch16 (native batch layout),rhs=1`): `jax-cpu` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 24.3x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.073 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.084 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 81.2x faster than the slowest successful cell (`jax-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 27.5x faster than the slowest successful cell (`jax-cpu`, 0.245 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 22.5x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 71.7x faster than the slowest successful cell (`tenferro-eager`, 0.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 13.3x faster than the slowest successful cell (`tenferro-eager`, 0.607 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 50.3x faster than the slowest successful cell (`pytorch-cpu`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.477 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 27.9x faster than the slowest successful cell (`tenferro-eager`, 0.498 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 30.0x faster than the slowest successful cell (`pytorch-cpu`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 42.7x faster than the slowest successful cell (`pytorch-cpu`, 0.392 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.392 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 48.5x faster than the slowest successful cell (`pytorch-cpu`, 0.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.845 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 71.2x faster than the slowest successful cell (`tenferro-eager`, 1.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 22.1x faster than the slowest successful cell (`tenferro-eager`, 1.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 79.7x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.797 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 50.8x faster than the slowest successful cell (`tenferro-eager`, 0.697 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.9x faster than the slowest successful cell (`pytorch-cpu`, 0.194 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 25.0x faster than the slowest successful cell (`pytorch-cpu`, 0.184 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.9x faster than the slowest successful cell (`pytorch-cpu`, 0.203 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 64.9x faster than the slowest successful cell (`pytorch-cpu`, 0.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 20.9x faster than the slowest successful cell (`pytorch-cpu`, 0.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_vjp` (f64, threads=4, shape=`8x8,rhs=1`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

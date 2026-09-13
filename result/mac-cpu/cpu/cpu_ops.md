# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `mac-cpu`
- Timestamp: `20260913_124035`

Original baseline run: `./scripts/run_all.sh 4`.

Original 4-thread baseline is under `data/results/mac-cpu/cpu/einsum/20260913_124035`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

## Thread Environment

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Targeted timing correction

- Refresh: `20260913_130307`, Apple M5 Max, 1 and 4 threads; tenferro `a48866b1a0bb52e6f9712c485925105b14ea30b9` (unchanged).
- Collection commands: `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/collection.sh` and `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/view_recheck.sh`.
- Sampling: 3 warmups and 15 measured runs; sequential collection with the idle-host guard enabled.
- Raw corrected rows and per-thread metadata: `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/`.
- All unselected cells are retained verbatim from the original runs named below; the tables combine those original measurements with this targeted correction.

- Replaced only tenferro-eager CPU ops rows (194 per thread). Runtime initialization occurs before sampling and the runtime is shared. Input construction, eager wrapping, operation execution and output consumption remain timed, as in the original suite.
- Baselines: `data/results/mac-cpu/cpu/einsum/20260913_123040/` (1T) and `data/results/mac-cpu/cpu/einsum/20260913_124035/` (4T). Trace and Python cells were not rerun.

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | 12.225 ± 0.177 | 11.399 ± 0.158 | 47.735 ± 0.284 | 39.086 ± 1.388 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | 0.204 ± 0.001 | 0.186 ± 0.002 | 0.758 ± 0.029 | 0.852 ± 0.018 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | 3.122 ± 0.084 | 2.845 ± 0.039 | 12.080 ± 0.163 | 9.973 ± 0.516 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | 0.794 ± 0.013 | 0.719 ± 0.033 | 3.009 ± 0.077 | 2.668 ± 0.049 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | 1.008 ± 0.018 | 1.021 ± 0.068 | 4.191 ± 0.083 | 4.258 ± 0.027 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.025 ± 0.000 | 0.029 ± 0.002 | 0.071 ± 0.001 | 0.093 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | 0.262 ± 0.004 | 0.264 ± 0.007 | 1.062 ± 0.013 | 1.116 ± 0.024 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.073 ± 0.003 | 0.074 ± 0.004 | 0.262 ± 0.003 | 0.316 ± 0.009 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | 1.644 ± 0.053 | 1.622 ± 0.097 | 6.441 ± 0.067 | 6.298 ± 0.243 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.033 ± 0.002 | 0.038 ± 0.001 | 0.104 ± 0.001 | 0.125 ± 0.004 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | 0.424 ± 0.010 | 0.408 ± 0.008 | 1.614 ± 0.038 | 1.656 ± 0.030 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.112 ± 0.001 | 0.112 ± 0.001 | 0.399 ± 0.010 | 0.451 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | 3.845 ± 0.048 | 3.684 ± 0.071 | 14.238 ± 0.128 | 11.630 ± 0.284 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | 0.065 ± 0.001 | 0.069 ± 0.001 | 0.224 ± 0.004 | 0.267 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | 1.034 ± 0.289 | 0.913 ± 0.027 | 3.540 ± 0.083 | 3.298 ± 0.098 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | 0.250 ± 0.006 | 0.240 ± 0.010 | 0.874 ± 0.028 | 0.909 ± 0.030 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 12.523 ± 0.235 | 11.112 ± 0.058 | 49.115 ± 0.153 | 37.011 ± 1.375 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.220 ± 0.010 | 0.208 ± 0.004 | 0.772 ± 0.060 | 0.800 ± 0.118 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 3.088 ± 0.075 | 2.808 ± 0.028 | 12.039 ± 0.489 | 9.894 ± 0.207 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.811 ± 0.035 | 0.725 ± 0.010 | 3.116 ± 0.193 | 2.642 ± 0.197 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.017 ± 0.046 | 0.997 ± 0.009 | 4.307 ± 0.105 | 4.504 ± 0.116 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.046 ± 0.003 | 0.044 ± 0.003 | 0.072 ± 0.004 | 0.094 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.271 ± 0.017 | 0.277 ± 0.003 | 1.065 ± 0.024 | 1.169 ± 0.058 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.094 ± 0.006 | 0.092 ± 0.002 | 0.273 ± 0.009 | 0.330 ± 0.054 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 1.639 ± 0.081 | 1.580 ± 0.015 | 6.429 ± 0.072 | 6.612 ± 0.233 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.052 ± 0.005 | 0.059 ± 0.010 | 0.108 ± 0.009 | 0.129 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.438 ± 0.020 | 0.421 ± 0.009 | 1.655 ± 0.085 | 1.737 ± 0.074 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.131 ± 0.009 | 0.131 ± 0.002 | 0.408 ± 0.017 | 0.473 ± 0.046 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 3.861 ± 0.093 | 3.551 ± 0.029 | 14.124 ± 0.237 | 11.899 ± 0.236 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.086 ± 0.004 | 0.087 ± 0.005 | 0.224 ± 0.005 | 0.297 ± 0.035 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 1.012 ± 0.037 | 0.909 ± 0.026 | 3.665 ± 0.200 | 3.400 ± 0.244 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.266 ± 0.023 | 0.251 ± 0.003 | 0.893 ± 0.041 | 0.954 ± 0.083 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | 0.486 ± 0.032 | 0.325 ± 0.024 | 55.998 ± 0.409 | 53.846 ± 1.627 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | 0.020 ± 0.002 | 0.021 ± 0.001 | 0.948 ± 0.056 | 0.956 ± 0.057 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | 0.135 ± 0.005 | 0.099 ± 0.007 | 13.965 ± 0.300 | 13.094 ± 0.481 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | 0.044 ± 0.001 | 0.036 ± 0.001 | 3.491 ± 0.048 | 3.337 ± 0.157 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | 0.066 ± 0.007 | 0.049 ± 0.001 | 0.856 ± 0.018 | 0.923 ± 0.042 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.025 ± 0.001 | 0.107 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | 0.023 ± 0.001 | 0.024 ± 0.001 | 0.227 ± 0.004 | 0.328 ± 0.022 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.067 ± 0.003 | 0.155 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | 0.077 ± 0.006 | 0.061 ± 0.003 | 3.432 ± 0.052 | 3.356 ± 0.066 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.011 ± 0.001 | 0.016 ± 0.001 | 0.063 ± 0.002 | 0.161 ± 0.029 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | 0.029 ± 0.000 | 0.027 ± 0.002 | 0.868 ± 0.013 | 0.963 ± 0.020 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.226 ± 0.003 | 0.324 ± 0.025 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | 0.171 ± 0.012 | 0.126 ± 0.004 | 14.178 ± 0.247 | 13.101 ± 0.447 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | 0.013 ± 0.001 | 0.017 ± 0.000 | 0.232 ± 0.006 | 0.316 ± 0.013 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | 0.057 ± 0.001 | 0.044 ± 0.001 | 3.506 ± 0.058 | 3.397 ± 0.117 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | 0.022 ± 0.004 | 0.022 ± 0.001 | 0.897 ± 0.034 | 0.985 ± 0.096 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 0.502 ± 0.034 | 0.326 ± 0.023 | 53.979 ± 1.012 | 53.277 ± 1.223 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.048 ± 0.003 | 0.048 ± 0.002 | 0.881 ± 0.083 | 1.017 ± 0.076 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.178 ± 0.013 | 0.124 ± 0.013 | 14.771 ± 0.625 | 13.667 ± 0.365 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.070 ± 0.006 | 0.062 ± 0.005 | 3.475 ± 0.318 | 3.578 ± 0.138 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 0.071 ± 0.008 | 0.079 ± 0.003 | 0.874 ± 0.017 | 0.973 ± 0.095 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.040 ± 0.008 | 0.048 ± 0.003 | 0.025 ± 0.002 | 0.111 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.046 ± 0.004 | 0.051 ± 0.011 | 0.226 ± 0.001 | 0.334 ± 0.037 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.040 ± 0.004 | 0.046 ± 0.002 | 0.065 ± 0.001 | 0.164 ± 0.011 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 0.096 ± 0.005 | 0.091 ± 0.005 | 3.341 ± 0.042 | 3.644 ± 0.108 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.043 ± 0.013 | 0.045 ± 0.002 | 0.067 ± 0.003 | 0.162 ± 0.008 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.051 ± 0.003 | 0.054 ± 0.001 | 0.868 ± 0.044 | 0.986 ± 0.117 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.043 ± 0.004 | 0.047 ± 0.002 | 0.236 ± 0.022 | 0.373 ± 0.061 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 0.191 ± 0.007 | 0.153 ± 0.004 | 13.746 ± 0.158 | 13.559 ± 0.294 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.042 ± 0.006 | 0.046 ± 0.003 | 0.239 ± 0.012 | 0.351 ± 0.069 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.082 ± 0.004 | 0.073 ± 0.003 | 3.472 ± 0.104 | 3.557 ± 0.168 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.047 ± 0.003 | 0.051 ± 0.003 | 0.891 ± 0.020 | 1.014 ± 0.067 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | 3.197 ± 0.043 | 3.085 ± 0.084 | 34.775 ± 0.705 | 31.126 ± 1.056 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | 0.057 ± 0.003 | 0.062 ± 0.001 | 0.533 ± 0.019 | 0.566 ± 0.022 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | 0.809 ± 0.020 | 0.761 ± 0.034 | 8.645 ± 0.062 | 8.050 ± 0.084 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | 0.217 ± 0.010 | 0.202 ± 0.006 | 2.128 ± 0.049 | 2.175 ± 0.078 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | 1.035 ± 0.026 | 1.101 ± 0.082 | 2.820 ± 0.105 | 2.786 ± 0.062 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.025 ± 0.000 | 0.031 ± 0.001 | 0.050 ± 0.002 | 0.067 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | 0.267 ± 0.005 | 0.275 ± 0.007 | 0.709 ± 0.012 | 0.745 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.074 ± 0.003 | 0.078 ± 0.003 | 0.179 ± 0.006 | 0.201 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | 1.228 ± 0.029 | 1.247 ± 0.085 | 4.574 ± 0.052 | 4.469 ± 0.055 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.027 ± 0.004 | 0.033 ± 0.001 | 0.075 ± 0.001 | 0.094 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | 0.316 ± 0.002 | 0.320 ± 0.013 | 1.129 ± 0.010 | 1.167 ± 0.071 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.086 ± 0.003 | 0.089 ± 0.003 | 0.285 ± 0.007 | 0.325 ± 0.009 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | 1.752 ± 0.028 | 1.710 ± 0.049 | 10.570 ± 0.070 | 10.018 ± 0.091 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | 0.035 ± 0.002 | 0.042 ± 0.001 | 0.166 ± 0.005 | 0.207 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | 0.455 ± 0.004 | 0.438 ± 0.004 | 2.608 ± 0.063 | 2.612 ± 0.025 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | 0.120 ± 0.001 | 0.120 ± 0.001 | 0.655 ± 0.012 | 0.691 ± 0.024 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 3.188 ± 0.162 | 2.945 ± 0.038 | 33.918 ± 0.234 | 32.795 ± 2.374 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.078 ± 0.005 | 0.078 ± 0.002 | 0.562 ± 0.033 | 0.598 ± 0.083 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.820 ± 0.031 | 0.766 ± 0.006 | 9.108 ± 0.173 | 8.475 ± 0.187 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.230 ± 0.020 | 0.219 ± 0.007 | 2.195 ± 0.232 | 2.214 ± 0.079 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.056 ± 0.050 | 1.054 ± 0.012 | 2.882 ± 0.078 | 2.939 ± 0.128 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.045 ± 0.003 | 0.054 ± 0.011 | 0.071 ± 0.028 | 0.070 ± 0.016 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.279 ± 0.008 | 0.290 ± 0.005 | 0.716 ± 0.018 | 0.768 ± 0.071 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.095 ± 0.006 | 0.100 ± 0.004 | 0.192 ± 0.014 | 0.202 ± 0.015 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 1.248 ± 0.028 | 1.240 ± 0.021 | 4.451 ± 0.115 | 4.657 ± 0.162 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.050 ± 0.014 | 0.052 ± 0.003 | 0.094 ± 0.007 | 0.095 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.325 ± 0.026 | 0.335 ± 0.007 | 1.187 ± 0.096 | 1.201 ± 0.050 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.109 ± 0.003 | 0.110 ± 0.005 | 0.294 ± 0.025 | 0.335 ± 0.017 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 1.739 ± 0.077 | 1.698 ± 0.025 | 10.541 ± 0.541 | 10.429 ± 0.256 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.061 ± 0.004 | 0.063 ± 0.005 | 0.189 ± 0.018 | 0.212 ± 0.023 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.472 ± 0.022 | 0.452 ± 0.015 | 2.665 ± 0.077 | 2.763 ± 0.154 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.135 ± 0.003 | 0.137 ± 0.002 | 0.670 ± 0.021 | 0.725 ± 0.038 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | 3.307 ± 0.086 | 2.799 ± 0.042 | 39.717 ± 0.359 | 39.632 ± 1.211 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | 0.063 ± 0.003 | 0.079 ± 0.011 | 0.638 ± 0.066 | 0.704 ± 0.023 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | 0.856 ± 0.016 | 0.711 ± 0.015 | 10.229 ± 0.082 | 10.128 ± 0.144 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | 0.226 ± 0.010 | 0.202 ± 0.004 | 2.513 ± 0.079 | 2.628 ± 0.043 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | 1.030 ± 0.028 | 1.999 ± 0.072 | 4.395 ± 0.209 | 4.249 ± 0.124 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.029 ± 0.000 | 0.061 ± 0.001 | 0.082 ± 0.003 | 0.110 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | 0.271 ± 0.001 | 0.524 ± 0.012 | 1.120 ± 0.037 | 1.155 ± 0.068 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.078 ± 0.001 | 0.155 ± 0.007 | 0.290 ± 0.019 | 0.334 ± 0.015 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | 1.149 ± 0.016 | 2.119 ± 0.138 | 6.272 ± 0.174 | 6.062 ± 0.155 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.031 ± 0.001 | 0.062 ± 0.001 | 0.111 ± 0.004 | 0.140 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | 0.300 ± 0.004 | 0.531 ± 0.006 | 1.556 ± 0.021 | 1.583 ± 0.034 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.085 ± 0.001 | 0.155 ± 0.005 | 0.396 ± 0.012 | 0.450 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | 1.628 ± 0.059 | 2.248 ± 0.035 | 12.667 ± 0.102 | 12.305 ± 0.287 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | 0.036 ± 0.001 | 0.065 ± 0.001 | 0.206 ± 0.006 | 0.257 ± 0.013 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | 0.553 ± 0.245 | 0.581 ± 0.012 | 3.161 ± 0.080 | 3.109 ± 0.061 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | 0.115 ± 0.008 | 0.168 ± 0.005 | 0.784 ± 0.014 | 0.840 ± 0.014 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | 3.439 ± 0.166 | 2.758 ± 0.045 | 41.076 ± 0.607 | 38.691 ± 1.563 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | 0.094 ± 0.005 | 0.125 ± 0.002 | 0.678 ± 0.026 | 0.720 ± 0.086 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | 0.884 ± 0.045 | 0.764 ± 0.011 | 10.623 ± 0.224 | 9.909 ± 0.428 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | 0.251 ± 0.010 | 0.264 ± 0.009 | 2.566 ± 0.157 | 2.628 ± 0.163 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | 1.072 ± 0.048 | 2.047 ± 0.034 | 4.368 ± 0.097 | 4.601 ± 0.253 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.057 ± 0.005 | 0.119 ± 0.005 | 0.094 ± 0.007 | 0.111 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | 0.307 ± 0.025 | 0.580 ± 0.013 | 1.137 ± 0.041 | 1.188 ± 0.135 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.102 ± 0.002 | 0.207 ± 0.006 | 0.303 ± 0.018 | 0.369 ± 0.093 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | 1.154 ± 0.032 | 2.097 ± 0.021 | 6.160 ± 0.071 | 6.406 ± 0.142 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.054 ± 0.003 | 0.115 ± 0.004 | 0.133 ± 0.005 | 0.140 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | 0.324 ± 0.020 | 0.594 ± 0.014 | 1.633 ± 0.048 | 1.647 ± 0.127 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.115 ± 0.008 | 0.217 ± 0.011 | 0.421 ± 0.022 | 0.463 ± 0.042 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | 1.617 ± 0.067 | 2.259 ± 0.038 | 12.442 ± 0.142 | 12.570 ± 0.477 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | 0.068 ± 0.011 | 0.119 ± 0.004 | 0.231 ± 0.031 | 0.261 ± 0.029 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | 0.448 ± 0.022 | 0.634 ± 0.013 | 3.241 ± 0.374 | 3.260 ± 0.099 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | 0.140 ± 0.006 | 0.225 ± 0.006 | 0.797 ± 0.026 | 0.918 ± 0.050 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | 22.029 ± 0.438 | 23.987 ± 1.213 | 53.161 ± 1.846 | 33.831 ± 0.791 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | 0.359 ± 0.015 | 0.364 ± 0.007 | 0.831 ± 0.034 | 0.792 ± 0.050 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | 5.630 ± 0.090 | 5.567 ± 0.021 | 13.358 ± 0.149 | 8.302 ± 0.401 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | 1.422 ± 0.066 | 1.400 ± 0.023 | 3.319 ± 0.070 | 2.339 ± 0.153 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | 1.630 ± 0.047 | 1.704 ± 0.106 | 3.070 ± 0.061 | 3.674 ± 0.063 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.036 ± 0.003 | 0.043 ± 0.002 | 0.056 ± 0.006 | 0.082 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | 0.415 ± 0.005 | 0.418 ± 0.005 | 0.757 ± 0.013 | 0.979 ± 0.048 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.112 ± 0.002 | 0.116 ± 0.002 | 0.198 ± 0.016 | 0.259 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | 2.769 ± 0.062 | 2.864 ± 0.078 | 5.775 ± 0.179 | 5.167 ± 0.050 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.050 ± 0.001 | 0.060 ± 0.002 | 0.093 ± 0.003 | 0.122 ± 0.004 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | 0.698 ± 0.016 | 0.690 ± 0.006 | 1.453 ± 0.055 | 1.627 ± 0.048 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.184 ± 0.008 | 0.183 ± 0.003 | 0.365 ± 0.018 | 0.437 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | 6.698 ± 0.212 | 6.799 ± 0.268 | 15.238 ± 0.158 | 10.215 ± 0.084 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | 0.111 ± 0.009 | 0.123 ± 0.009 | 0.255 ± 0.017 | 0.284 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | 1.703 ± 0.032 | 1.684 ± 0.028 | 3.813 ± 0.091 | 2.921 ± 0.099 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | 0.429 ± 0.010 | 0.445 ± 0.037 | 0.935 ± 0.034 | 1.010 ± 0.022 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 22.250 ± 0.274 | 21.938 ± 0.073 | 52.608 ± 0.130 | 33.895 ± 1.685 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.379 ± 0.013 | 0.379 ± 0.004 | 0.842 ± 0.074 | 0.866 ± 0.072 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 5.590 ± 0.057 | 5.513 ± 0.021 | 14.298 ± 0.368 | 8.706 ± 0.266 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 1.396 ± 0.057 | 1.403 ± 0.017 | 3.337 ± 0.177 | 2.385 ± 0.116 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 1.621 ± 0.056 | 1.625 ± 0.024 | 3.117 ± 0.179 | 3.934 ± 0.161 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.058 ± 0.004 | 0.065 ± 0.006 | 0.055 ± 0.003 | 0.087 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.434 ± 0.027 | 0.435 ± 0.008 | 0.772 ± 0.022 | 1.020 ± 0.086 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.130 ± 0.008 | 0.140 ± 0.002 | 0.196 ± 0.007 | 0.259 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 2.722 ± 0.085 | 2.701 ± 0.052 | 5.680 ± 0.079 | 5.425 ± 0.261 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.075 ± 0.008 | 0.083 ± 0.006 | 0.097 ± 0.003 | 0.128 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.697 ± 0.050 | 0.707 ± 0.021 | 1.465 ± 0.054 | 1.703 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.196 ± 0.008 | 0.204 ± 0.007 | 0.361 ± 0.010 | 0.456 ± 0.028 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 6.688 ± 0.113 | 6.591 ± 0.038 | 15.080 ± 0.088 | 10.516 ± 0.353 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.146 ± 0.012 | 0.145 ± 0.004 | 0.244 ± 0.014 | 0.308 ± 0.078 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 1.694 ± 0.096 | 1.681 ± 0.010 | 3.814 ± 0.052 | 3.055 ± 0.179 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.458 ± 0.027 | 0.451 ± 0.010 | 0.957 ± 0.029 | 1.055 ± 0.068 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | 1.374 ± 0.153 | 0.705 ± 0.018 | 58.113 ± 1.368 | 53.076 ± 1.083 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | 0.107 ± 0.015 | 0.053 ± 0.001 | 0.963 ± 0.025 | 1.302 ± 0.112 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | 0.383 ± 0.006 | 0.217 ± 0.019 | 14.677 ± 0.081 | 13.486 ± 0.198 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | 0.160 ± 0.002 | 0.086 ± 0.003 | 3.680 ± 0.077 | 3.876 ± 0.168 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | 0.182 ± 0.011 | 0.107 ± 0.001 | 0.910 ± 0.049 | 1.325 ± 0.082 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.077 ± 0.001 | 0.038 ± 0.001 | 0.050 ± 0.002 | 0.471 ± 0.015 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | 0.105 ± 0.001 | 0.057 ± 0.015 | 0.260 ± 0.006 | 0.678 ± 0.026 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.084 ± 0.007 | 0.041 ± 0.002 | 0.091 ± 0.001 | 0.508 ± 0.014 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | 0.244 ± 0.004 | 0.147 ± 0.017 | 3.494 ± 0.110 | 3.787 ± 0.219 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.078 ± 0.001 | 0.038 ± 0.001 | 0.097 ± 0.003 | 0.516 ± 0.039 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | 0.121 ± 0.002 | 0.064 ± 0.001 | 0.891 ± 0.015 | 1.347 ± 0.073 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.088 ± 0.006 | 0.043 ± 0.001 | 0.252 ± 0.004 | 0.669 ± 0.021 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | 0.680 ± 0.017 | 0.487 ± 0.024 | 17.346 ± 0.770 | 13.272 ± 0.107 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | 0.095 ± 0.004 | 0.047 ± 0.001 | 0.307 ± 0.007 | 0.680 ± 0.046 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | 0.235 ± 0.005 | 0.150 ± 0.008 | 4.326 ± 0.072 | 3.799 ± 0.115 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | 0.120 ± 0.003 | 0.066 ± 0.002 | 1.101 ± 0.045 | 1.312 ± 0.067 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | 1.398 ± 0.142 | 0.700 ± 0.050 | 57.333 ± 0.589 | 52.972 ± 1.174 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | 0.226 ± 0.019 | 0.118 ± 0.008 | 0.951 ± 0.041 | 1.353 ± 0.078 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | 0.553 ± 0.023 | 0.290 ± 0.006 | 14.814 ± 2.781 | 14.193 ± 0.333 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | 0.296 ± 0.039 | 0.150 ± 0.004 | 3.699 ± 0.232 | 3.981 ± 0.290 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | 0.321 ± 0.010 | 0.177 ± 0.009 | 0.901 ± 0.027 | 1.352 ± 0.160 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.212 ± 0.021 | 0.115 ± 0.005 | 0.052 ± 0.002 | 0.479 ± 0.043 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | 0.225 ± 0.029 | 0.125 ± 0.011 | 0.258 ± 0.012 | 0.682 ± 0.034 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.207 ± 0.012 | 0.114 ± 0.020 | 0.092 ± 0.004 | 0.522 ± 0.039 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | 0.385 ± 0.018 | 0.212 ± 0.004 | 3.434 ± 0.037 | 4.074 ± 0.239 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.210 ± 0.017 | 0.107 ± 0.004 | 0.096 ± 0.011 | 0.532 ± 0.034 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | 0.256 ± 0.019 | 0.130 ± 0.007 | 0.992 ± 0.115 | 1.452 ± 0.135 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.226 ± 0.018 | 0.115 ± 0.009 | 0.259 ± 0.009 | 0.718 ± 0.065 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | 0.835 ± 0.052 | 0.527 ± 0.033 | 16.562 ± 0.111 | 14.149 ± 0.391 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | 0.222 ± 0.017 | 0.115 ± 0.004 | 0.307 ± 0.008 | 0.703 ± 0.055 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | 0.384 ± 0.009 | 0.224 ± 0.013 | 4.438 ± 0.411 | 4.106 ± 0.215 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | 0.250 ± 0.013 | 0.133 ± 0.024 | 1.109 ± 0.062 | 1.389 ± 0.203 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | 10.214 ± 0.172 | 5.238 ± 0.070 | 41.067 ± 0.179 | 40.560 ± 1.210 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | 0.269 ± 0.003 | 0.172 ± 0.005 | 0.668 ± 0.032 | 1.042 ± 0.028 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | 2.673 ± 0.056 | 1.402 ± 0.155 | 10.386 ± 0.198 | 10.389 ± 0.715 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | 0.757 ± 0.017 | 0.413 ± 0.012 | 2.571 ± 0.103 | 2.932 ± 0.208 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | 4.240 ± 0.121 | 3.788 ± 0.122 | 4.502 ± 0.120 | 4.672 ± 0.100 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.172 ± 0.003 | 0.148 ± 0.005 | 0.100 ± 0.001 | 0.467 ± 0.028 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | 1.143 ± 0.031 | 1.008 ± 0.040 | 1.116 ± 0.019 | 1.606 ± 0.331 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.369 ± 0.005 | 0.321 ± 0.025 | 0.312 ± 0.028 | 0.674 ± 0.046 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | 4.542 ± 0.127 | 4.085 ± 0.041 | 6.423 ± 0.124 | 6.432 ± 0.099 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.175 ± 0.007 | 0.148 ± 0.004 | 0.131 ± 0.003 | 0.490 ± 0.015 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | 1.228 ± 0.014 | 1.040 ± 0.032 | 1.619 ± 0.062 | 1.942 ± 0.058 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.395 ± 0.033 | 0.325 ± 0.044 | 0.420 ± 0.013 | 0.797 ± 0.053 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | 5.788 ± 0.036 | 4.210 ± 0.061 | 13.114 ± 0.744 | 12.344 ± 0.318 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | 0.191 ± 0.011 | 0.156 ± 0.002 | 0.226 ± 0.006 | 0.591 ± 0.029 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | 1.554 ± 0.067 | 1.088 ± 0.033 | 3.217 ± 0.156 | 3.439 ± 0.110 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | 0.466 ± 0.015 | 0.347 ± 0.021 | 0.803 ± 0.019 | 1.166 ± 0.066 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | 10.546 ± 0.435 | 5.178 ± 0.083 | 41.599 ± 0.682 | 39.698 ± 1.325 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | 0.464 ± 0.013 | 0.326 ± 0.010 | 0.721 ± 0.041 | 1.056 ± 0.068 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | 2.878 ± 0.079 | 1.528 ± 0.045 | 10.683 ± 0.288 | 10.655 ± 0.495 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | 0.911 ± 0.059 | 0.575 ± 0.010 | 2.646 ± 0.175 | 2.974 ± 0.167 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | 4.340 ± 0.142 | 3.821 ± 0.048 | 4.426 ± 0.084 | 5.093 ± 0.130 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.362 ± 0.013 | 0.303 ± 0.012 | 0.128 ± 0.020 | 0.496 ± 0.100 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | 1.320 ± 0.049 | 1.157 ± 0.022 | 1.173 ± 0.045 | 1.542 ± 0.092 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.556 ± 0.047 | 0.481 ± 0.020 | 0.321 ± 0.010 | 0.733 ± 0.084 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | 4.762 ± 0.048 | 3.978 ± 0.057 | 6.277 ± 0.095 | 6.948 ± 0.142 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.395 ± 0.204 | 0.309 ± 0.012 | 0.155 ± 0.013 | 0.513 ± 0.032 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | 1.381 ± 0.086 | 1.201 ± 0.058 | 1.659 ± 0.081 | 2.071 ± 0.186 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.572 ± 0.025 | 0.495 ± 0.013 | 0.445 ± 0.026 | 0.805 ± 0.067 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | 5.862 ± 0.119 | 4.243 ± 0.036 | 12.676 ± 0.065 | 13.175 ± 0.287 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | 0.408 ± 0.046 | 0.308 ± 0.017 | 0.267 ± 0.037 | 0.601 ± 0.023 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | 1.713 ± 0.077 | 1.271 ± 0.025 | 3.320 ± 0.485 | 3.708 ± 0.243 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | 0.640 ± 0.019 | 0.496 ± 0.011 | 0.851 ± 0.040 | 1.221 ± 0.102 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | 1.163 ± 0.096 | 0.638 ± 0.021 | 2.301 ± 0.079 | 2.399 ± 0.047 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | 8.772 ± 0.298 | 2.749 ± 0.087 | 9.720 ± 0.080 | 9.374 ± 0.418 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | 0.256 ± 0.029 | 0.211 ± 0.005 | 0.648 ± 0.017 | 0.690 ± 0.015 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | 1.164 ± 0.024 | 0.661 ± 0.016 | 2.351 ± 0.048 | 2.470 ± 0.088 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | 8.978 ± 1.708 | 2.747 ± 0.063 | 9.223 ± 0.085 | 9.800 ± 0.440 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | 0.297 ± 0.010 | 0.237 ± 0.011 | 0.640 ± 0.013 | 0.724 ± 0.062 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | 79.144 ± 2.247 | 85.153 ± 1.650 | 75.746 ± 1.608 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 3.102 ± 0.107 | 3.177 ± 0.134 | 3.353 ± 0.105 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 14.011 ± 0.067 | 15.173 ± 0.141 | 14.101 ± 0.179 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | 79.280 ± 2.002 | 84.966 ± 3.913 | 75.984 ± 0.875 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 3.144 ± 0.065 | 3.200 ± 0.103 | 3.550 ± 0.118 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 13.649 ± 0.086 | 15.826 ± 0.941 | 14.546 ± 0.091 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | 77.603 ± 4.109 | 73.993 ± 1.750 | 72.659 ± 0.542 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 3.052 ± 0.050 | 2.924 ± 0.100 | 3.376 ± 0.066 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 13.937 ± 0.096 | 13.563 ± 0.138 | 14.117 ± 0.172 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | 76.930 ± 3.692 | 72.221 ± 2.275 | 74.733 ± 0.309 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 3.166 ± 0.054 | 2.897 ± 0.098 | 3.610 ± 0.109 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 13.624 ± 0.094 | 13.733 ± 1.050 | 14.557 ± 0.297 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | 31.773 ± 1.322 | 65.929 ± 2.641 | 20.353 ± 0.309 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 0.737 ± 0.060 | 1.405 ± 0.094 | 0.984 ± 0.043 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 4.317 ± 0.244 | 9.034 ± 0.063 | 3.727 ± 0.098 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | 32.616 ± 2.297 | 61.391 ± 4.324 | 21.579 ± 0.473 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 0.925 ± 0.048 | 1.411 ± 0.115 | 1.063 ± 0.093 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 4.474 ± 0.163 | 8.170 ± 0.104 | 3.981 ± 0.145 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | 32.199 ± 0.885 | 33.929 ± 0.950 | 22.446 ± 0.202 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 0.765 ± 0.095 | 0.782 ± 0.048 | 1.100 ± 0.101 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 4.349 ± 0.146 | 5.011 ± 0.232 | 3.933 ± 0.165 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | 32.081 ± 1.996 | 31.195 ± 1.988 | 23.571 ± 0.811 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 0.888 ± 0.129 | 0.894 ± 0.037 | 1.291 ± 0.095 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 4.242 ± 0.055 | 4.526 ± 0.068 | 4.220 ± 0.174 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | 0.049 ± 0.005 | 0.038 ± 0.002 | 3.270 ± 0.105 | 3.428 ± 0.109 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.023 ± 0.002 | 0.026 ± 0.001 | 0.872 ± 0.023 | 0.892 ± 0.031 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | 0.074 ± 0.005 | 0.080 ± 0.004 | 3.325 ± 0.043 | 3.504 ± 0.127 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.051 ± 0.004 | 0.068 ± 0.013 | 0.862 ± 0.022 | 0.918 ± 0.069 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | 0.160 ± 0.005 | 0.076 ± 0.000 | 3.469 ± 0.081 | 3.756 ± 0.150 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.096 ± 0.006 | 0.048 ± 0.001 | 0.856 ± 0.049 | 1.274 ± 0.124 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | 0.276 ± 0.017 | 0.148 ± 0.006 | 3.301 ± 0.054 | 3.906 ± 0.210 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.264 ± 0.043 | 0.117 ± 0.007 | 0.870 ± 0.026 | 1.244 ± 0.110 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | 53.302 ± 3.853 | 59.613 ± 1.262 | 37.574 ± 1.187 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 2.108 ± 0.159 | 2.023 ± 0.081 | 1.938 ± 0.085 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 9.141 ± 0.061 | 10.572 ± 0.217 | 7.402 ± 0.661 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | 51.691 ± 3.883 | 54.421 ± 1.679 | 39.112 ± 1.415 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 2.142 ± 0.089 | 1.962 ± 0.055 | 2.138 ± 0.117 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 8.495 ± 0.080 | 8.671 ± 0.049 | 7.692 ± 0.188 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | 52.023 ± 0.528 | 59.176 ± 1.580 | 43.825 ± 0.571 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 2.112 ± 0.054 | 1.913 ± 0.076 | 2.148 ± 0.071 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 9.249 ± 0.121 | 10.039 ± 0.188 | 8.407 ± 0.246 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | 59.124 ± 3.427 | 57.644 ± 5.196 | 44.912 ± 0.885 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 2.161 ± 0.053 | 1.836 ± 0.123 | 2.269 ± 0.196 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 8.353 ± 0.139 | 8.349 ± 0.120 | 8.811 ± 0.236 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | 0.919 ± 0.025 | 0.165 ± 0.002 | 1.850 ± 0.074 | 2.168 ± 0.165 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 0.253 ± 0.015 | 0.109 ± 0.002 | 0.510 ± 0.018 | 0.858 ± 0.029 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | 1.094 ± 0.039 | 0.323 ± 0.009 | 1.773 ± 0.066 | 2.270 ± 0.169 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 0.439 ± 0.031 | 0.265 ± 0.013 | 0.500 ± 0.009 | 0.881 ± 0.061 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | 6.217 ± 0.187 | 6.840 ± 0.133 | 5.283 ± 0.197 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 0.318 ± 0.016 | 0.353 ± 0.013 | 0.407 ± 0.058 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 1.268 ± 0.050 | 1.436 ± 0.045 | 1.268 ± 0.077 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | 5.921 ± 0.125 | 6.496 ± 0.539 | 6.133 ± 0.302 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 0.458 ± 0.049 | 0.375 ± 0.021 | 0.436 ± 0.073 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 1.303 ± 0.017 | 1.219 ± 0.069 | 1.431 ± 0.125 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | 6.142 ± 0.308 | 10.686 ± 0.222 | 5.468 ± 0.145 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 0.315 ± 0.011 | 0.440 ± 0.040 | 0.449 ± 0.011 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 1.233 ± 0.026 | 2.096 ± 0.151 | 1.454 ± 0.081 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | 5.826 ± 0.197 | 10.538 ± 0.375 | 6.183 ± 0.240 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.454 ± 0.046 | 0.476 ± 0.037 | 0.464 ± 0.106 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 1.267 ± 0.012 | 1.856 ± 0.081 | 1.480 ± 0.207 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | 2.619 ± 0.090 | 1.320 ± 0.051 | 2.994 ± 0.059 | 3.261 ± 0.096 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 0.625 ± 0.026 | 0.330 ± 0.002 | 0.747 ± 0.029 | 0.905 ± 0.031 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | 2.674 ± 0.083 | 1.344 ± 0.023 | 3.011 ± 0.086 | 3.454 ± 0.172 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 0.763 ± 0.035 | 0.395 ± 0.019 | 0.738 ± 0.020 | 0.943 ± 0.169 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | 118.889 ± 0.399 | 141.574 ± 2.291 | 121.574 ± 0.786 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 5.351 ± 0.086 | 5.747 ± 0.126 | 5.845 ± 0.114 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 23.403 ± 0.088 | 26.797 ± 0.136 | 24.446 ± 0.172 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | 122.109 ± 3.053 | 139.852 ± 6.233 | 126.327 ± 0.903 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 5.249 ± 0.057 | 5.728 ± 0.063 | 6.100 ± 0.200 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 23.984 ± 0.462 | 25.698 ± 0.155 | 25.496 ± 0.212 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | 118.751 ± 1.679 | 117.795 ± 23.144 | 116.342 ± 1.709 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 5.353 ± 0.171 | 5.212 ± 0.148 | 5.592 ± 0.071 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 23.314 ± 0.129 | 23.200 ± 0.188 | 23.526 ± 0.151 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | 121.107 ± 2.146 | 117.767 ± 9.731 | 120.394 ± 1.236 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 5.215 ± 0.047 | 5.087 ± 0.077 | 5.876 ± 0.105 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 23.120 ± 0.048 | 22.887 ± 0.104 | 24.450 ± 0.318 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | 8.202 ± 0.596 | 5.683 ± 0.598 | 229.608 ± 3.553 | 220.555 ± 4.098 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | 0.040 ± 0.002 | 0.034 ± 0.000 | 3.447 ± 0.068 | 3.356 ± 0.095 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | 0.160 ± 0.007 | 0.115 ± 0.004 | 14.010 ± 0.211 | 13.246 ± 0.322 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | 0.937 ± 0.119 | 0.679 ± 0.020 | 56.062 ± 2.348 | 53.448 ± 1.428 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | 6.703 ± 0.350 | 4.791 ± 0.176 | 221.597 ± 3.518 | 218.644 ± 10.092 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | 0.064 ± 0.006 | 0.062 ± 0.004 | 3.279 ± 0.041 | 3.542 ± 0.267 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | 0.193 ± 0.010 | 0.145 ± 0.002 | 13.608 ± 0.264 | 13.802 ± 0.773 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | 0.915 ± 0.075 | 0.664 ± 0.014 | 53.453 ± 0.992 | 53.556 ± 2.083 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | 1.796 ± 0.097 | 1.335 ± 0.036 | 55.814 ± 0.966 | 55.076 ± 1.366 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 0.604 ± 0.019 | 0.383 ± 0.006 | 54.877 ± 0.727 | 53.646 ± 2.572 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | 1.642 ± 0.101 | 1.310 ± 0.009 | 54.136 ± 1.027 | 54.022 ± 0.555 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 0.562 ± 0.044 | 0.377 ± 0.005 | 53.337 ± 0.750 | 57.706 ± 4.062 | - | - |
| large | `qr` | f64 | 1 | `128x128` | 0.401 ± 0.009 | 0.394 ± 0.008 | 2.135 ± 0.060 | 2.083 ± 0.086 | - | - |
| large | `qr` | f64 | 1 | `256x256` | 1.363 ± 0.170 | 1.180 ± 0.018 | 8.268 ± 0.120 | 7.934 ± 0.219 | - | - |
| large | `qr` | f64 | 1 | `64x64` | 0.086 ± 0.011 | 0.082 ± 0.004 | 0.498 ± 0.016 | 0.543 ± 0.015 | - | - |
| large | `qr` | f64 | 4 | `128x128` | 0.422 ± 0.010 | 0.406 ± 0.012 | 2.001 ± 0.033 | 2.237 ± 0.069 | - | - |
| large | `qr` | f64 | 4 | `256x256` | 1.276 ± 0.040 | 1.205 ± 0.041 | 7.881 ± 0.035 | 7.946 ± 0.279 | - | - |
| large | `qr` | f64 | 4 | `64x64` | 0.099 ± 0.005 | 0.095 ± 0.005 | 0.523 ± 0.021 | 0.560 ± 0.027 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | 0.612 ± 0.059 | 0.081 ± 0.002 | 1.854 ± 0.049 | 1.794 ± 0.119 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | 0.607 ± 0.048 | 0.084 ± 0.001 | 2.061 ± 0.055 | 1.978 ± 0.064 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | 0.622 ± 0.053 | 0.101 ± 0.002 | 2.724 ± 0.063 | 2.654 ± 0.114 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | 6.131 ± 0.257 | 0.227 ± 0.014 | 7.229 ± 0.168 | 7.001 ± 0.178 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | 5.773 ± 0.350 | 0.230 ± 0.007 | 7.650 ± 0.091 | 7.111 ± 0.316 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | 5.824 ± 0.194 | 0.286 ± 0.022 | 8.929 ± 0.094 | 8.781 ± 0.256 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.081 ± 0.002 | 0.045 ± 0.004 | 0.484 ± 0.014 | 0.531 ± 0.015 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.083 ± 0.004 | 0.052 ± 0.000 | 0.583 ± 0.015 | 0.621 ± 0.018 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.094 ± 0.005 | 0.058 ± 0.003 | 0.974 ± 0.046 | 0.919 ± 0.031 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | 0.601 ± 0.035 | 0.135 ± 0.006 | 1.711 ± 0.026 | 1.888 ± 0.112 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | 0.610 ± 0.026 | 0.138 ± 0.004 | 1.901 ± 0.050 | 2.082 ± 0.164 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | 0.635 ± 0.027 | 0.150 ± 0.007 | 2.505 ± 0.043 | 2.759 ± 0.144 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | 5.483 ± 0.088 | 0.298 ± 0.021 | 6.943 ± 0.045 | 7.134 ± 0.223 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | 5.658 ± 0.079 | 0.294 ± 0.020 | 7.358 ± 0.074 | 7.739 ± 0.229 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | 5.636 ± 0.176 | 0.343 ± 0.011 | 8.598 ± 0.042 | 9.010 ± 0.206 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.114 ± 0.012 | 0.099 ± 0.008 | 0.465 ± 0.009 | 0.537 ± 0.030 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.113 ± 0.003 | 0.099 ± 0.009 | 0.567 ± 0.016 | 0.637 ± 0.053 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.125 ± 0.016 | 0.107 ± 0.008 | 0.884 ± 0.034 | 1.001 ± 0.194 | - | - |
| large | `svd` | f64 | 1 | `128x128` | 1.269 ± 0.087 | 1.232 ± 0.017 | 2.954 ± 0.120 | 2.951 ± 0.066 | - | - |
| large | `svd` | f64 | 1 | `256x256` | 5.572 ± 0.457 | 4.886 ± 0.025 | 11.831 ± 0.083 | 11.322 ± 0.184 | - | - |
| large | `svd` | f64 | 1 | `64x64` | 0.297 ± 0.018 | 0.289 ± 0.005 | 0.736 ± 0.053 | 0.753 ± 0.022 | - | - |
| large | `svd` | f64 | 4 | `128x128` | 1.267 ± 0.054 | 1.257 ± 0.014 | 2.897 ± 0.032 | 3.121 ± 0.268 | - | - |
| large | `svd` | f64 | 4 | `256x256` | 5.059 ± 0.048 | 4.968 ± 0.073 | 11.297 ± 0.065 | 11.970 ± 0.310 | - | - |
| large | `svd` | f64 | 4 | `64x64` | 0.322 ± 0.011 | 0.317 ± 0.011 | 0.726 ± 0.036 | 0.807 ± 0.074 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | 0.024 ± 0.000 | 0.021 ± 0.001 | 0.052 ± 0.001 | 0.068 ± 0.005 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | 0.011 ± 0.003 | 0.011 ± 0.001 | 0.009 ± 0.000 | 0.029 ± 0.007 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | 0.062 ± 0.011 | 0.050 ± 0.001 | 0.164 ± 0.003 | 0.194 ± 0.008 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.001 | 0.029 ± 0.006 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.018 ± 0.000 | 0.036 ± 0.007 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | 0.039 ± 0.003 | 0.041 ± 0.005 | 0.051 ± 0.001 | 0.067 ± 0.004 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | 0.032 ± 0.005 | 0.032 ± 0.004 | 0.009 ± 0.000 | 0.029 ± 0.005 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | 0.074 ± 0.007 | 0.065 ± 0.003 | 0.159 ± 0.004 | 0.234 ± 0.032 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | 0.029 ± 0.005 | 0.037 ± 0.011 | 0.010 ± 0.000 | 0.028 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | 0.032 ± 0.003 | 0.036 ± 0.004 | 0.017 ± 0.001 | 0.035 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | 0.015 ± 0.000 | 0.017 ± 0.002 | 0.068 ± 0.002 | 0.160 ± 0.017 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.014 ± 0.001 | 0.015 ± 0.001 | 0.013 ± 0.001 | 0.100 ± 0.008 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | 0.016 ± 0.002 | 0.019 ± 0.002 | 0.249 ± 0.026 | 0.325 ± 0.017 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.104 ± 0.014 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.003 | 0.120 ± 0.014 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | 0.042 ± 0.004 | 0.030 ± 0.003 | 0.069 ± 0.001 | 0.159 ± 0.015 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.043 ± 0.006 | 0.032 ± 0.004 | 0.012 ± 0.001 | 0.116 ± 0.017 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | 0.042 ± 0.006 | 0.030 ± 0.004 | 0.230 ± 0.007 | 0.322 ± 0.016 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.049 ± 0.010 | 0.033 ± 0.008 | 0.014 ± 0.000 | 0.095 ± 0.014 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.038 ± 0.004 | 0.034 ± 0.003 | 0.025 ± 0.001 | 0.118 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | 0.077 ± 0.002 | 0.049 ± 0.002 | 0.132 ± 0.007 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.060 ± 0.001 | 0.036 ± 0.002 | 0.127 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | 0.107 ± 0.001 | 0.075 ± 0.003 | 0.169 ± 0.017 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.062 ± 0.001 | 0.037 ± 0.001 | 0.136 ± 0.011 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.068 ± 0.003 | 0.047 ± 0.007 | 0.130 ± 0.004 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | 0.192 ± 0.007 | 0.051 ± 0.003 | 0.129 ± 0.009 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.188 ± 0.027 | 0.036 ± 0.003 | 0.130 ± 0.016 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | 0.222 ± 0.014 | 0.075 ± 0.002 | 0.181 ± 0.022 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.189 ± 0.019 | 0.034 ± 0.001 | 0.129 ± 0.006 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.183 ± 0.009 | 0.043 ± 0.002 | 0.130 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | 0.077 ± 0.001 | 0.047 ± 0.008 | 0.318 ± 0.038 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.061 ± 0.001 | 0.031 ± 0.001 | 0.314 ± 0.021 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | 0.102 ± 0.003 | 0.072 ± 0.020 | 0.347 ± 0.040 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.063 ± 0.001 | 0.034 ± 0.002 | 0.313 ± 0.009 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.069 ± 0.001 | 0.038 ± 0.001 | 0.317 ± 0.005 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | 0.181 ± 0.005 | 0.046 ± 0.004 | 0.326 ± 0.016 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.173 ± 0.011 | 0.033 ± 0.005 | 0.307 ± 0.016 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | 0.216 ± 0.010 | 0.075 ± 0.004 | 0.331 ± 0.033 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.174 ± 0.017 | 0.030 ± 0.001 | 0.315 ± 0.013 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.178 ± 0.010 | 0.039 ± 0.001 | 0.319 ± 0.012 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | 0.068 ± 0.005 | 0.060 ± 0.003 | 0.189 ± 0.007 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.061 ± 0.002 | 0.053 ± 0.003 | 0.185 ± 0.005 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | 0.080 ± 0.001 | 0.074 ± 0.003 | 0.196 ± 0.006 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.060 ± 0.001 | 0.054 ± 0.003 | 0.189 ± 0.008 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.062 ± 0.004 | 0.057 ± 0.002 | 0.186 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | 0.194 ± 0.013 | 0.139 ± 0.008 | 0.187 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.187 ± 0.042 | 0.134 ± 0.010 | 0.181 ± 0.009 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | 0.204 ± 0.010 | 0.150 ± 0.006 | 0.203 ± 0.099 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.193 ± 0.024 | 0.133 ± 0.007 | 0.188 ± 0.012 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.184 ± 0.010 | 0.139 ± 0.019 | 0.191 ± 0.023 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | 0.086 ± 0.001 | 0.046 ± 0.001 | 0.515 ± 0.021 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.078 ± 0.001 | 0.043 ± 0.002 | 0.517 ± 0.061 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | 0.101 ± 0.002 | 0.057 ± 0.004 | 0.530 ± 0.027 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.081 ± 0.008 | 0.043 ± 0.001 | 0.504 ± 0.029 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.084 ± 0.002 | 0.048 ± 0.001 | 0.507 ± 0.021 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | 0.191 ± 0.007 | 0.102 ± 0.006 | 0.512 ± 0.024 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.192 ± 0.014 | 0.103 ± 0.014 | 0.513 ± 0.017 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | 0.211 ± 0.005 | 0.114 ± 0.012 | 0.570 ± 0.184 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.195 ± 0.014 | 0.106 ± 0.025 | 0.499 ± 0.015 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.195 ± 0.013 | 0.103 ± 0.004 | 0.517 ± 0.043 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | 0.093 ± 0.005 | 0.042 ± 0.001 | 0.079 ± 0.004 | 0.421 ± 0.035 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.096 ± 0.009 | 0.035 ± 0.000 | 0.022 ± 0.002 | 0.350 ± 0.012 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | 0.096 ± 0.002 | 0.043 ± 0.000 | 0.242 ± 0.003 | 0.581 ± 0.046 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.089 ± 0.001 | 0.036 ± 0.001 | 0.023 ± 0.003 | 0.365 ± 0.011 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.092 ± 0.002 | 0.041 ± 0.001 | 0.035 ± 0.001 | 0.390 ± 0.035 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | 0.214 ± 0.032 | 0.106 ± 0.008 | 0.078 ± 0.002 | 0.419 ± 0.015 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.206 ± 0.021 | 0.110 ± 0.011 | 0.020 ± 0.001 | 0.350 ± 0.017 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | 0.212 ± 0.021 | 0.105 ± 0.006 | 0.240 ± 0.012 | 0.632 ± 0.079 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.202 ± 0.020 | 0.109 ± 0.016 | 0.023 ± 0.002 | 0.363 ± 0.019 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.218 ± 0.017 | 0.107 ± 0.007 | 0.039 ± 0.003 | 0.379 ± 0.018 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | 0.153 ± 0.003 | 0.048 ± 0.002 | 0.188 ± 0.005 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.148 ± 0.007 | 0.042 ± 0.001 | 0.183 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | 0.170 ± 0.003 | 0.059 ± 0.004 | 0.192 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.143 ± 0.004 | 0.045 ± 0.008 | 0.187 ± 0.009 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.141 ± 0.002 | 0.044 ± 0.002 | 0.189 ± 0.008 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | 0.350 ± 0.022 | 0.083 ± 0.005 | 0.186 ± 0.004 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.351 ± 0.039 | 0.103 ± 0.025 | 0.184 ± 0.012 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | 0.404 ± 0.030 | 0.089 ± 0.009 | 0.193 ± 0.007 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.339 ± 0.016 | 0.073 ± 0.011 | 0.184 ± 0.029 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.350 ± 0.017 | 0.081 ± 0.006 | 0.192 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | 0.177 ± 0.008 | 0.045 ± 0.004 | 0.466 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.160 ± 0.002 | 0.043 ± 0.002 | 0.459 ± 0.009 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | 0.185 ± 0.004 | 0.056 ± 0.001 | 0.477 ± 0.018 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.165 ± 0.006 | 0.042 ± 0.002 | 0.468 ± 0.012 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.169 ± 0.002 | 0.043 ± 0.002 | 0.468 ± 0.014 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | 0.368 ± 0.010 | 0.085 ± 0.009 | 0.471 ± 0.044 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.378 ± 0.052 | 0.080 ± 0.015 | 0.466 ± 0.017 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | 0.401 ± 0.024 | 0.093 ± 0.021 | 0.525 ± 0.152 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.365 ± 0.015 | 0.075 ± 0.009 | 0.458 ± 0.021 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.364 ± 0.013 | 0.073 ± 0.013 | 0.482 ± 0.034 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | 0.128 ± 0.008 | 0.083 ± 0.001 | 0.067 ± 0.001 | 0.428 ± 0.032 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 0.127 ± 0.004 | 0.081 ± 0.001 | 0.032 ± 0.001 | 0.397 ± 0.021 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | 0.149 ± 0.005 | 0.097 ± 0.003 | 0.163 ± 0.005 | 0.534 ± 0.068 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.129 ± 0.004 | 0.081 ± 0.001 | 0.033 ± 0.001 | 0.402 ± 0.019 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.130 ± 0.002 | 0.083 ± 0.006 | 0.038 ± 0.001 | 0.399 ± 0.034 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | 0.297 ± 0.022 | 0.236 ± 0.010 | 0.066 ± 0.002 | 0.437 ± 0.035 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.292 ± 0.015 | 0.242 ± 0.014 | 0.031 ± 0.002 | 0.386 ± 0.011 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | 0.323 ± 0.024 | 0.238 ± 0.018 | 0.157 ± 0.004 | 0.549 ± 0.122 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.301 ± 0.029 | 0.241 ± 0.017 | 0.032 ± 0.003 | 0.389 ± 0.017 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.316 ± 0.077 | 0.236 ± 0.010 | 0.040 ± 0.003 | 0.390 ± 0.007 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | 0.060 ± 0.002 | 0.096 ± 0.005 | 0.155 ± 0.011 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.059 ± 0.002 | 0.099 ± 0.023 | 0.157 ± 0.006 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | 0.067 ± 0.000 | 0.103 ± 0.004 | 0.160 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.055 ± 0.006 | 0.099 ± 0.010 | 0.157 ± 0.013 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.059 ± 0.003 | 0.097 ± 0.012 | 0.149 ± 0.005 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | 0.176 ± 0.007 | 0.095 ± 0.005 | 0.164 ± 0.014 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.181 ± 0.019 | 0.093 ± 0.004 | 0.154 ± 0.009 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | 0.189 ± 0.010 | 0.105 ± 0.006 | 0.162 ± 0.006 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.180 ± 0.011 | 0.095 ± 0.006 | 0.150 ± 0.011 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.180 ± 0.013 | 0.094 ± 0.003 | 0.151 ± 0.007 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | 0.064 ± 0.002 | 0.049 ± 0.003 | 0.363 ± 0.027 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.063 ± 0.000 | 0.044 ± 0.002 | 0.355 ± 0.023 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | 0.078 ± 0.001 | 0.058 ± 0.007 | 0.360 ± 0.036 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.062 ± 0.001 | 0.046 ± 0.003 | 0.342 ± 0.019 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.063 ± 0.000 | 0.048 ± 0.005 | 0.343 ± 0.019 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | 0.173 ± 0.011 | 0.050 ± 0.003 | 0.353 ± 0.032 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.179 ± 0.011 | 0.047 ± 0.003 | 0.347 ± 0.011 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | 0.182 ± 0.004 | 0.065 ± 0.004 | 0.352 ± 0.016 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.177 ± 0.011 | 0.046 ± 0.002 | 0.335 ± 0.008 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.176 ± 0.005 | 0.047 ± 0.003 | 0.336 ± 0.020 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | 0.133 ± 0.003 | 0.073 ± 0.001 | 0.071 ± 0.003 | 0.418 ± 0.011 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 0.085 ± 0.016 | 0.048 ± 0.001 | 0.021 ± 0.001 | 0.385 ± 0.013 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | 0.212 ± 0.009 | 0.120 ± 0.001 | 0.199 ± 0.005 | 0.503 ± 0.016 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 0.083 ± 0.008 | 0.049 ± 0.001 | 0.025 ± 0.002 | 0.395 ± 0.022 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 0.092 ± 0.008 | 0.054 ± 0.001 | 0.033 ± 0.003 | 0.401 ± 0.015 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | 0.210 ± 0.014 | 0.130 ± 0.004 | 0.073 ± 0.003 | 0.418 ± 0.012 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 0.184 ± 0.011 | 0.106 ± 0.008 | 0.021 ± 0.002 | 0.395 ± 0.027 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | 0.309 ± 0.014 | 0.170 ± 0.005 | 0.200 ± 0.002 | 0.513 ± 0.042 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 0.184 ± 0.034 | 0.115 ± 0.013 | 0.022 ± 0.001 | 0.384 ± 0.007 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 0.176 ± 0.011 | 0.112 ± 0.008 | 0.033 ± 0.002 | 0.396 ± 0.034 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | 0.072 ± 0.001 | 0.075 ± 0.002 | 0.138 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.047 ± 0.004 | 0.052 ± 0.003 | 0.136 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | 0.112 ± 0.007 | 0.124 ± 0.009 | 0.191 ± 0.021 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.048 ± 0.000 | 0.052 ± 0.004 | 0.141 ± 0.015 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.057 ± 0.002 | 0.060 ± 0.001 | 0.133 ± 0.006 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | 0.143 ± 0.007 | 0.077 ± 0.002 | 0.146 ± 0.013 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.122 ± 0.011 | 0.051 ± 0.003 | 0.136 ± 0.018 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | 0.188 ± 0.007 | 0.125 ± 0.005 | 0.221 ± 0.045 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.126 ± 0.008 | 0.051 ± 0.002 | 0.136 ± 0.009 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.130 ± 0.008 | 0.061 ± 0.002 | 0.137 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | 0.073 ± 0.004 | 0.057 ± 0.002 | 0.342 ± 0.005 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.048 ± 0.000 | 0.038 ± 0.008 | 0.343 ± 0.011 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | 0.119 ± 0.002 | 0.106 ± 0.008 | 0.361 ± 0.032 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.049 ± 0.000 | 0.035 ± 0.001 | 0.347 ± 0.009 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.051 ± 0.003 | 0.038 ± 0.002 | 0.362 ± 0.033 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | 0.129 ± 0.005 | 0.058 ± 0.003 | 0.339 ± 0.022 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.108 ± 0.005 | 0.034 ± 0.007 | 0.341 ± 0.007 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | 0.172 ± 0.016 | 0.104 ± 0.002 | 0.361 ± 0.022 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.114 ± 0.016 | 0.036 ± 0.003 | 0.333 ± 0.014 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.111 ± 0.006 | 0.038 ± 0.002 | 0.350 ± 0.015 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | 0.014 ± 0.000 | 0.016 ± 0.001 | 0.064 ± 0.002 | 0.104 ± 0.009 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | 0.015 ± 0.003 | 0.014 ± 0.000 | 0.009 ± 0.001 | 0.039 ± 0.005 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | 0.017 ± 0.000 | 0.019 ± 0.002 | 0.243 ± 0.019 | 0.261 ± 0.017 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.010 ± 0.000 | 0.041 ± 0.004 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | 0.013 ± 0.001 | 0.014 ± 0.000 | 0.021 ± 0.005 | 0.061 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | 0.042 ± 0.008 | 0.046 ± 0.003 | 0.064 ± 0.003 | 0.102 ± 0.004 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | 0.044 ± 0.008 | 0.051 ± 0.006 | 0.007 ± 0.001 | 0.044 ± 0.006 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | 0.041 ± 0.006 | 0.044 ± 0.005 | 0.221 ± 0.001 | 0.268 ± 0.013 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | 0.036 ± 0.005 | 0.045 ± 0.002 | 0.010 ± 0.000 | 0.036 ± 0.002 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | 0.039 ± 0.007 | 0.046 ± 0.006 | 0.020 ± 0.001 | 0.062 ± 0.003 | - | - |
| small | `qr` | f64 | 1 | `16x16` | 0.013 ± 0.000 | 0.015 ± 0.001 | 0.038 ± 0.002 | 0.053 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `2x2` | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.007 ± 0.000 | 0.030 ± 0.007 | - | - |
| small | `qr` | f64 | 1 | `32x32` | 0.018 ± 0.000 | 0.020 ± 0.001 | 0.125 ± 0.004 | 0.161 ± 0.008 | - | - |
| small | `qr` | f64 | 1 | `4x4` | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.008 ± 0.000 | 0.026 ± 0.004 | - | - |
| small | `qr` | f64 | 1 | `8x8` | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.014 ± 0.001 | 0.031 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `16x16` | 0.030 ± 0.003 | 0.036 ± 0.006 | 0.049 ± 0.003 | 0.052 ± 0.004 | - | - |
| small | `qr` | f64 | 4 | `2x2` | 0.031 ± 0.006 | 0.036 ± 0.002 | 0.022 ± 0.008 | 0.029 ± 0.011 | - | - |
| small | `qr` | f64 | 4 | `32x32` | 0.042 ± 0.010 | 0.044 ± 0.003 | 0.136 ± 0.010 | 0.161 ± 0.005 | - | - |
| small | `qr` | f64 | 4 | `4x4` | 0.029 ± 0.005 | 0.034 ± 0.005 | 0.022 ± 0.001 | 0.026 ± 0.006 | - | - |
| small | `qr` | f64 | 4 | `8x8` | 0.031 ± 0.003 | 0.034 ± 0.004 | 0.031 ± 0.002 | 0.029 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | 0.018 ± 0.000 | 0.029 ± 0.000 | 0.051 ± 0.001 | 0.077 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | 0.018 ± 0.000 | 0.029 ± 0.000 | 0.056 ± 0.003 | 0.081 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.016 ± 0.002 | 0.026 ± 0.001 | 0.015 ± 0.001 | 0.040 ± 0.003 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.015 ± 0.000 | 0.028 ± 0.001 | 0.016 ± 0.000 | 0.039 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | 0.030 ± 0.000 | 0.036 ± 0.000 | 0.146 ± 0.009 | 0.185 ± 0.016 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | 0.030 ± 0.003 | 0.036 ± 0.000 | 0.148 ± 0.004 | 0.190 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.015 ± 0.002 | 0.028 ± 0.000 | 0.017 ± 0.001 | 0.042 ± 0.010 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.015 ± 0.000 | 0.028 ± 0.001 | 0.017 ± 0.000 | 0.043 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.016 ± 0.001 | 0.028 ± 0.000 | 0.022 ± 0.000 | 0.046 ± 0.004 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.016 ± 0.000 | 0.028 ± 0.000 | 0.025 ± 0.000 | 0.050 ± 0.004 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | 0.045 ± 0.006 | 0.083 ± 0.002 | 0.050 ± 0.001 | 0.075 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | 0.044 ± 0.004 | 0.081 ± 0.005 | 0.055 ± 0.001 | 0.081 ± 0.006 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.043 ± 0.007 | 0.087 ± 0.008 | 0.014 ± 0.001 | 0.037 ± 0.003 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.039 ± 0.005 | 0.082 ± 0.007 | 0.014 ± 0.000 | 0.041 ± 0.010 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | 0.053 ± 0.004 | 0.087 ± 0.010 | 0.135 ± 0.002 | 0.205 ± 0.075 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | 0.051 ± 0.002 | 0.081 ± 0.009 | 0.149 ± 0.002 | 0.192 ± 0.045 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.042 ± 0.002 | 0.085 ± 0.009 | 0.015 ± 0.001 | 0.039 ± 0.005 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.041 ± 0.003 | 0.081 ± 0.007 | 0.017 ± 0.001 | 0.043 ± 0.008 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.040 ± 0.003 | 0.080 ± 0.003 | 0.021 ± 0.000 | 0.054 ± 0.013 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.041 ± 0.004 | 0.081 ± 0.008 | 0.023 ± 0.001 | 0.047 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `16x16` | 0.038 ± 0.001 | 0.034 ± 0.001 | 0.058 ± 0.004 | 0.072 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `2x2` | 0.013 ± 0.001 | 0.016 ± 0.002 | 0.009 ± 0.001 | 0.035 ± 0.009 | - | - |
| small | `svd` | f64 | 1 | `32x32` | 0.091 ± 0.001 | 0.084 ± 0.004 | 0.184 ± 0.003 | 0.223 ± 0.015 | - | - |
| small | `svd` | f64 | 1 | `4x4` | 0.014 ± 0.001 | 0.017 ± 0.000 | 0.011 ± 0.001 | 0.029 ± 0.004 | - | - |
| small | `svd` | f64 | 1 | `8x8` | 0.019 ± 0.000 | 0.020 ± 0.001 | 0.020 ± 0.001 | 0.037 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `16x16` | 0.050 ± 0.004 | 0.056 ± 0.006 | 0.056 ± 0.001 | 0.069 ± 0.003 | - | - |
| small | `svd` | f64 | 4 | `2x2` | 0.033 ± 0.009 | 0.042 ± 0.004 | 0.008 ± 0.001 | 0.040 ± 0.016 | - | - |
| small | `svd` | f64 | 4 | `32x32` | 0.104 ± 0.014 | 0.103 ± 0.011 | 0.180 ± 0.003 | 0.213 ± 0.005 | - | - |
| small | `svd` | f64 | 4 | `4x4` | 0.035 ± 0.010 | 0.041 ± 0.005 | 0.011 ± 0.000 | 0.027 ± 0.004 | - | - |
| small | `svd` | f64 | 4 | `8x8` | 0.037 ± 0.003 | 0.042 ± 0.003 | 0.020 ± 0.001 | 0.035 ± 0.003 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 115.3x faster than the slowest successful cell (`pytorch-cpu`, 55.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 172.5x faster than the slowest successful cell (`pytorch-cpu`, 55.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 47.9x faster than the slowest successful cell (`jax-cpu`, 0.956 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 46.7x faster than the slowest successful cell (`jax-cpu`, 0.956 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 103.5x faster than the slowest successful cell (`pytorch-cpu`, 13.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 141.2x faster than the slowest successful cell (`pytorch-cpu`, 13.965 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 79.1x faster than the slowest successful cell (`pytorch-cpu`, 3.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 97.5x faster than the slowest successful cell (`pytorch-cpu`, 3.491 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-eager` is 13.9x faster than the slowest successful cell (`jax-cpu`, 0.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`jax-cpu`, 0.923 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`jax-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 44.5x faster than the slowest successful cell (`pytorch-cpu`, 3.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 56.4x faster than the slowest successful cell (`pytorch-cpu`, 3.432 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 15.1x faster than the slowest successful cell (`jax-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-eager` is 33.3x faster than the slowest successful cell (`jax-cpu`, 0.963 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 35.9x faster than the slowest successful cell (`jax-cpu`, 0.963 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 21.3x faster than the slowest successful cell (`jax-cpu`, 0.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 82.8x faster than the slowest successful cell (`pytorch-cpu`, 14.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 112.7x faster than the slowest successful cell (`pytorch-cpu`, 14.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.316 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 61.7x faster than the slowest successful cell (`pytorch-cpu`, 3.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 79.4x faster than the slowest successful cell (`pytorch-cpu`, 3.506 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 45.0x faster than the slowest successful cell (`jax-cpu`, 0.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 44.3x faster than the slowest successful cell (`jax-cpu`, 0.985 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 107.6x faster than the slowest successful cell (`pytorch-cpu`, 53.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 165.6x faster than the slowest successful cell (`pytorch-cpu`, 53.979 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 21.2x faster than the slowest successful cell (`jax-cpu`, 1.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 1.017 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 82.9x faster than the slowest successful cell (`pytorch-cpu`, 14.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 118.7x faster than the slowest successful cell (`pytorch-cpu`, 14.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 51.1x faster than the slowest successful cell (`jax-cpu`, 3.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 57.4x faster than the slowest successful cell (`jax-cpu`, 3.578 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-eager` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.973 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 37.9x faster than the slowest successful cell (`jax-cpu`, 3.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 39.9x faster than the slowest successful cell (`jax-cpu`, 3.644 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `tenferro-eager` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.986 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 71.9x faster than the slowest successful cell (`pytorch-cpu`, 13.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 89.7x faster than the slowest successful cell (`pytorch-cpu`, 13.746 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 43.4x faster than the slowest successful cell (`jax-cpu`, 3.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 48.7x faster than the slowest successful cell (`jax-cpu`, 3.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 21.8x faster than the slowest successful cell (`jax-cpu`, 1.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`jax-cpu`, 1.014 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 34.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 34.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 8.645 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 8.645 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 10.0x faster than the slowest successful cell (`jax-cpu`, 2.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`jax-cpu`, 2.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 33.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 33.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 9.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 9.108 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_qr` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 2.214 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 39.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 39.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch16 (native batch layout),rhs=1`): `tenferro-eager` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.704 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 10.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 10.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 11.6x faster than the slowest successful cell (`jax-cpu`, 2.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=1, shape=`16x16xbatch64 (native batch layout),rhs=1`): `tenferro-trace` is 13.0x faster than the slowest successful cell (`jax-cpu`, 2.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-eager` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 41.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout),rhs=1`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 41.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 10.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`pytorch-cpu`, 10.623 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 10.5x faster than the slowest successful cell (`jax-cpu`, 2.628 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 42.3x faster than the slowest successful cell (`pytorch-cpu`, 58.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 82.5x faster than the slowest successful cell (`pytorch-cpu`, 58.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 12.1x faster than the slowest successful cell (`jax-cpu`, 1.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 24.7x faster than the slowest successful cell (`jax-cpu`, 1.302 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 38.3x faster than the slowest successful cell (`pytorch-cpu`, 14.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 67.7x faster than the slowest successful cell (`pytorch-cpu`, 14.677 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 24.3x faster than the slowest successful cell (`jax-cpu`, 3.876 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 45.1x faster than the slowest successful cell (`jax-cpu`, 3.876 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 1.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch256 (native batch layout)`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.678 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.508 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 3.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 25.7x faster than the slowest successful cell (`jax-cpu`, 3.787 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.516 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-eager` is 11.1x faster than the slowest successful cell (`jax-cpu`, 1.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`jax-cpu`, 1.347 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.669 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 17.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 35.6x faster than the slowest successful cell (`pytorch-cpu`, 17.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.680 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 4.326 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 28.8x faster than the slowest successful cell (`pytorch-cpu`, 4.326 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 1.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 19.7x faster than the slowest successful cell (`jax-cpu`, 1.312 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 41.0x faster than the slowest successful cell (`pytorch-cpu`, 57.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 81.9x faster than the slowest successful cell (`pytorch-cpu`, 57.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 1.353 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 26.8x faster than the slowest successful cell (`pytorch-cpu`, 14.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 51.2x faster than the slowest successful cell (`pytorch-cpu`, 14.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 13.5x faster than the slowest successful cell (`jax-cpu`, 3.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 26.5x faster than the slowest successful cell (`jax-cpu`, 3.981 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 10.6x faster than the slowest successful cell (`jax-cpu`, 4.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 19.2x faster than the slowest successful cell (`jax-cpu`, 4.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`4x4xbatch256 (native batch layout)`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 1.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 16.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 31.4x faster than the slowest successful cell (`pytorch-cpu`, 16.562 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 4.438 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 4.438 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`jax-cpu`, 1.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=1, shape=`128x128`): `tenferro-eager` is 69.8x faster than the slowest successful cell (`jax-cpu`, 3.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=1, shape=`128x128`): `tenferro-trace` is 91.2x faster than the slowest successful cell (`jax-cpu`, 3.428 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=1, shape=`64x64`): `tenferro-eager` is 39.3x faster than the slowest successful cell (`jax-cpu`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=1, shape=`64x64`): `tenferro-trace` is 33.7x faster than the slowest successful cell (`jax-cpu`, 0.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 47.5x faster than the slowest successful cell (`jax-cpu`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 43.6x faster than the slowest successful cell (`jax-cpu`, 3.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-eager` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.918 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=1, shape=`128x128`): `tenferro-eager` is 23.5x faster than the slowest successful cell (`jax-cpu`, 3.756 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=1, shape=`128x128`): `tenferro-trace` is 49.5x faster than the slowest successful cell (`jax-cpu`, 3.756 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=1, shape=`64x64`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`jax-cpu`, 1.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=1, shape=`64x64`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 1.274 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 14.2x faster than the slowest successful cell (`jax-cpu`, 3.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 26.4x faster than the slowest successful cell (`jax-cpu`, 3.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_matmul_backward` (f64, threads=4, shape=`64x64`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 1.244 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=1, shape=`128x128,rhs=1`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`jax-cpu`, 2.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`1024x1024`): `tenferro-eager` is 28.0x faster than the slowest successful cell (`pytorch-cpu`, 229.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`1024x1024`): `tenferro-trace` is 40.4x faster than the slowest successful cell (`pytorch-cpu`, 229.608 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`128x128`): `tenferro-eager` is 85.9x faster than the slowest successful cell (`pytorch-cpu`, 3.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`128x128`): `tenferro-trace` is 100.9x faster than the slowest successful cell (`pytorch-cpu`, 3.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`256x256`): `tenferro-eager` is 87.8x faster than the slowest successful cell (`pytorch-cpu`, 14.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`256x256`): `tenferro-trace` is 121.9x faster than the slowest successful cell (`pytorch-cpu`, 14.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`512x512`): `tenferro-eager` is 59.9x faster than the slowest successful cell (`pytorch-cpu`, 56.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=1, shape=`512x512`): `tenferro-trace` is 82.5x faster than the slowest successful cell (`pytorch-cpu`, 56.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`1024x1024`): `tenferro-eager` is 33.1x faster than the slowest successful cell (`pytorch-cpu`, 221.597 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 46.2x faster than the slowest successful cell (`pytorch-cpu`, 221.597 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-eager` is 55.3x faster than the slowest successful cell (`jax-cpu`, 3.542 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`128x128`): `tenferro-trace` is 57.0x faster than the slowest successful cell (`jax-cpu`, 3.542 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 71.6x faster than the slowest successful cell (`jax-cpu`, 13.802 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 94.9x faster than the slowest successful cell (`jax-cpu`, 13.802 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`512x512`): `tenferro-eager` is 58.5x faster than the slowest successful cell (`jax-cpu`, 53.556 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 80.6x faster than the slowest successful cell (`jax-cpu`, 53.556 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=1, shape=`1024x256 * 256x1024`): `tenferro-eager` is 31.1x faster than the slowest successful cell (`pytorch-cpu`, 55.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=1, shape=`1024x256 * 256x1024`): `tenferro-trace` is 41.8x faster than the slowest successful cell (`pytorch-cpu`, 55.814 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=1, shape=`256x1024 * 1024x256`): `tenferro-eager` is 90.9x faster than the slowest successful cell (`pytorch-cpu`, 54.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=1, shape=`256x1024 * 1024x256`): `tenferro-trace` is 143.2x faster than the slowest successful cell (`pytorch-cpu`, 54.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`1024x256 * 256x1024`): `tenferro-eager` is 33.0x faster than the slowest successful cell (`pytorch-cpu`, 54.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`1024x256 * 256x1024`): `tenferro-trace` is 41.3x faster than the slowest successful cell (`pytorch-cpu`, 54.136 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-eager` is 102.7x faster than the slowest successful cell (`jax-cpu`, 57.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/matmul_rect` (f64, threads=4, shape=`256x1024 * 1024x256`): `tenferro-trace` is 153.0x faster than the slowest successful cell (`jax-cpu`, 57.706 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`128x128,rhs=1`): `tenferro-trace` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 1.854 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`128x128,rhs=16`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`pytorch-cpu`, 2.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`128x128,rhs=64`): `tenferro-trace` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 2.724 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`256x256,rhs=1`): `tenferro-trace` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 7.229 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`256x256,rhs=16`): `tenferro-trace` is 33.2x faster than the slowest successful cell (`pytorch-cpu`, 7.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`256x256,rhs=64`): `tenferro-trace` is 31.2x faster than the slowest successful cell (`pytorch-cpu`, 8.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`64x64,rhs=1`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.531 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`64x64,rhs=16`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`64x64,rhs=64`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=1, shape=`64x64,rhs=64`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.974 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=1`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`jax-cpu`, 1.888 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`jax-cpu`, 2.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`jax-cpu`, 2.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 23.9x faster than the slowest successful cell (`jax-cpu`, 7.134 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 7.739 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`jax-cpu`, 9.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.314 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_vjp` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.515 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.517 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.504 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.421 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.581 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.365 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.390 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.350 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_matmul_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.363 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.466 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.459 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.468 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.468 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`2x2,rhs=1`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.397 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`4x4,rhs=1`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.402 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=1, shape=`8x8,rhs=1`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.399 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`2x2,rhs=1`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.386 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_backward` (f64, threads=4, shape=`4x4,rhs=1`): `pytorch-cpu` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.389 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.385 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.395 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.401 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.395 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 17.2x faster than the slowest successful cell (`jax-cpu`, 0.384 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_backward` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_vjp` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.341 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/matmul` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.261 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-43-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 4

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_131021/cpu_ops_t4_20261124_131021.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 8.391 ± 0.105 | 8.293 ± 0.035 | 27.139 ± 0.162 | 17.889 ± 0.367 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.134 ± 0.002 | 0.132 ± 0.001 | 0.412 ± 0.004 | 0.366 ± 0.016 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 2.080 ± 0.006 | 2.083 ± 0.006 | 6.823 ± 0.063 | 10.546 ± 3.404 | - | - |
| batched | `batched_eigh` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.519 ± 0.001 | 0.519 ± 0.026 | 1.698 ± 0.024 | 1.140 ± 0.060 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.542 ± 0.001 | 0.540 ± 0.002 | 0.544 ± 0.006 | 0.428 ± 0.008 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.018 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.141 ± 0.001 | 0.140 ± 0.002 | 0.144 ± 0.004 | 0.112 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.043 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.726 ± 0.013 | 0.722 ± 0.003 | 2.702 ± 0.031 | 1.880 ± 0.466 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.049 ± 0.000 | 0.048 ± 0.000 | 0.050 ± 0.001 | 0.052 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.718 ± 0.007 | 0.713 ± 0.003 | 0.676 ± 0.002 | 0.656 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.185 ± 0.000 | 0.182 ± 0.001 | 0.176 ± 0.002 | 0.168 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 2.321 ± 0.004 | 2.205 ± 0.005 | 8.008 ± 0.121 | 4.580 ± 0.356 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.140 ± 0.001 | 0.139 ± 0.002 | 0.132 ± 0.001 | 0.146 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.560 ± 0.047 | 0.561 ± 0.001 | 2.000 ± 0.013 | 1.509 ± 0.222 | - | - |
| batched | `batched_eigh` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.145 ± 0.001 | 0.143 ± 0.000 | 0.506 ± 0.007 | 0.562 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 0.114 ± 0.001 | 0.113 ± 0.001 | 0.415 ± 0.020 | 1.587 ± 0.039 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.026 ± 0.001 | 0.163 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.041 ± 0.007 | 0.039 ± 0.001 | 0.107 ± 0.004 | 0.997 ± 0.192 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.043 ± 0.003 | 0.171 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.147 ± 0.001 | 0.148 ± 0.001 | 0.046 ± 0.002 | 0.279 ± 0.130 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.156 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.040 ± 0.000 | 0.041 ± 0.000 | 0.023 ± 0.001 | 0.157 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.155 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.045 ± 0.004 | 0.044 ± 0.000 | 0.072 ± 0.007 | 0.303 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.156 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.045 ± 0.002 | 0.044 ± 0.000 | 0.054 ± 0.001 | 0.224 ± 0.169 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.025 ± 0.001 | 0.157 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.049 ± 0.001 | 0.049 ± 0.000 | 0.097 ± 0.005 | 0.480 ± 0.009 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.000 | 0.156 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.017 ± 0.000 | 0.017 ± 0.000 | 0.040 ± 0.003 | 0.453 ± 0.307 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.036 ± 4.797 | 0.164 ± 0.011 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 2.446 ± 0.005 | 2.454 ± 0.071 | 5.603 ± 0.067 | 4.542 ± 0.073 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.046 ± 0.001 | 0.044 ± 0.001 | 0.100 ± 0.005 | 0.139 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.610 ± 0.002 | 0.616 ± 0.005 | 1.434 ± 0.023 | 4.557 ± 0.558 | - | - |
| batched | `batched_qr` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.159 ± 0.003 | 0.235 ± 0.002 | 0.370 ± 0.007 | 0.374 ± 0.019 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.319 ± 0.001 | 0.318 ± 0.002 | 0.231 ± 0.002 | 0.341 ± 0.043 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.085 ± 0.000 | 0.084 ± 0.001 | 0.064 ± 0.002 | 0.089 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.023 ± 0.001 | 0.032 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.210 ± 0.006 | 0.216 ± 0.003 | 0.657 ± 0.011 | 0.966 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.206 ± 0.002 | 0.205 ± 0.001 | 0.170 ± 0.002 | 0.246 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.055 ± 0.000 | 0.048 ± 0.002 | 0.066 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.797 ± 0.022 | 0.649 ± 0.188 | 1.653 ± 0.024 | 1.520 ± 0.211 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.047 ± 0.001 | 0.045 ± 0.000 | 0.034 ± 0.001 | 0.050 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.173 ± 0.002 | 0.167 ± 0.002 | 0.428 ± 0.008 | 0.823 ± 0.459 | - | - |
| batched | `batched_qr` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.168 ± 0.000 | 0.166 ± 0.000 | 0.114 ± 0.003 | 0.178 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.578 ± 0.010 | 0.653 ± 0.003 | 0.969 ± 0.071 | 2.266 ± 0.109 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.045 ± 0.000 | 0.044 ± 0.000 | 0.040 ± 0.003 | 0.062 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.154 ± 0.001 | 0.173 ± 0.002 | 0.263 ± 0.023 | 1.881 ± 0.305 | - | - |
| batched | `batched_solve` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.047 ± 0.000 | 0.053 ± 0.000 | 0.089 ± 0.007 | 0.203 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.212 ± 0.002 | 0.165 ± 0.000 | 0.088 ± 0.002 | 0.350 ± 0.102 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.017 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.059 ± 0.000 | 0.049 ± 0.000 | 0.038 ± 0.001 | 0.094 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.002 | 0.026 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.088 ± 0.001 | 0.116 ± 0.001 | 0.126 ± 0.001 | 0.498 ± 0.009 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.084 ± 0.000 | 0.076 ± 0.001 | 0.047 ± 0.001 | 0.241 ± 0.114 | - | - |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.027 ± 0.000 | 0.028 ± 0.076 | 0.040 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.218 ± 0.003 | 0.240 ± 0.003 | 0.354 ± 0.005 | 0.944 ± 0.177 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.062 ± 0.002 | 0.069 ± 0.000 | 0.113 ± 0.003 | 0.488 ± 0.109 | - | - |
| batched | `batched_solve` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.059 ± 0.001 | 0.057 ± 0.000 | 0.044 ± 0.002 | 0.075 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 16.400 ± 0.835 | 14.518 ± 0.053 | 60.115 ± 0.405 | 30.111 ± 0.920 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.231 ± 0.001 | 0.229 ± 0.001 | 0.922 ± 0.009 | 0.558 ± 0.092 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 3.622 ± 0.017 | 4.097 ± 0.103 | 15.135 ± 0.076 | 17.634 ± 3.163 | - | - |
| batched | `batched_svd` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.918 ± 0.073 | 0.907 ± 0.003 | 3.746 ± 0.021 | 1.953 ± 0.193 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 1.300 ± 0.004 | 1.305 ± 0.006 | 1.399 ± 0.034 | 1.588 ± 0.039 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.031 ± 0.000 | 0.038 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.333 ± 0.006 | 0.330 ± 0.003 | 0.354 ± 0.003 | 0.391 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.001 | 0.088 ± 0.001 | 0.099 ± 0.002 | 0.110 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 1.197 ± 0.011 | 1.188 ± 0.012 | 5.593 ± 0.049 | 2.439 ± 0.097 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.079 ± 0.001 | 0.078 ± 0.000 | 0.096 ± 0.001 | 0.092 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 1.177 ± 0.013 | 1.174 ± 0.003 | 1.395 ± 0.014 | 1.286 ± 0.019 | - | - |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.001 | 0.298 ± 0.000 | 0.357 ± 0.005 | 0.323 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 4.078 ± 0.014 | 3.842 ± 0.311 | 18.745 ± 0.075 | 8.139 ± 0.426 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.240 ± 0.000 | 0.238 ± 0.001 | 0.302 ± 0.004 | 0.261 ± 0.007 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.954 ± 0.003 | 0.957 ± 0.007 | 4.685 ± 0.018 | 3.212 ± 0.082 | - | - |
| batched | `batched_svd` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.244 ± 0.001 | 0.244 ± 0.003 | 1.190 ± 0.031 | 0.694 ± 0.044 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout)` | - | 1.372 ± 0.134 | 0.429 ± 0.005 | 10.505 ± 0.123 | 3.266 ± 0.082 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch16 (native batch layout)` | - | 0.097 ± 0.002 | 0.026 ± 0.000 | 0.263 ± 0.010 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch256 (native batch layout)` | - | 0.344 ± 0.005 | 0.121 ± 0.002 | 2.683 ± 0.084 | 1.393 ± 0.882 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `16x16xbatch64 (native batch layout)` | - | 0.166 ± 0.004 | 0.046 ± 0.000 | 0.716 ± 0.018 | 0.205 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout)` | - | 0.551 ± 0.004 | 0.478 ± 0.001 | 0.196 ± 0.003 | 0.026 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | - | 0.072 ± 0.001 | 0.023 ± 0.000 | 0.107 ± 0.002 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch256 (native batch layout)` | - | 0.183 ± 0.002 | 0.131 ± 0.000 | 0.127 ± 0.003 | 0.025 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | - | 0.094 ± 0.002 | 0.045 ± 0.000 | 0.110 ± 0.002 | 0.015 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout)` | - | 0.268 ± 0.003 | 0.151 ± 0.002 | 0.279 ± 0.009 | 0.816 ± 0.011 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | - | 0.072 ± 0.001 | 0.023 ± 0.000 | 0.112 ± 0.002 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch256 (native batch layout)` | - | 0.209 ± 0.003 | 0.139 ± 0.001 | 0.221 ± 0.006 | 0.391 ± 0.021 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | - | 0.101 ± 0.002 | 0.047 ± 0.000 | 0.137 ± 0.004 | 0.058 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout)` | - | 0.396 ± 0.005 | 0.169 ± 0.002 | 7.463 ± 0.063 | 1.330 ± 0.013 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch16 (native batch layout)` | - | 0.078 ± 0.000 | 0.026 ± 0.000 | 0.221 ± 0.013 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch256 (native batch layout)` | - | 0.184 ± 0.007 | 0.061 ± 0.000 | 1.927 ± 0.029 | 0.649 ± 0.040 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `8x8xbatch64 (native batch layout)` | - | 0.101 ± 0.001 | 0.029 ± 0.000 | 0.561 ± 0.026 | 0.090 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.292 ± 0.024 | 2.621 ± 0.064 | 2.061 ± 0.135 | 4.217 ± 0.181 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.147 ± 0.003 | 0.180 ± 0.001 | 0.128 ± 0.004 | 0.100 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.400 ± 0.008 | 0.706 ± 0.020 | 0.617 ± 0.028 | 2.247 ± 1.517 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.215 ± 0.001 | 0.304 ± 0.002 | 0.275 ± 0.015 | 0.350 ± 0.019 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.485 ± 0.012 | 0.784 ± 0.006 | 0.220 ± 0.007 | 0.770 ± 0.340 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.093 ± 0.002 | 0.062 ± 0.000 | 0.077 ± 0.001 | 0.028 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.183 ± 0.003 | 0.234 ± 0.001 | 0.115 ± 0.005 | 0.246 ± 0.120 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.113 ± 0.009 | 0.096 ± 0.000 | 0.087 ± 0.004 | 0.055 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.462 ± 0.006 | 0.683 ± 0.006 | 0.357 ± 0.006 | 1.014 ± 0.022 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.095 ± 0.001 | 0.067 ± 0.000 | 0.082 ± 0.002 | 0.025 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.223 ± 0.013 | 0.325 ± 0.001 | 0.148 ± 0.007 | 0.488 ± 0.079 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.124 ± 0.002 | 0.119 ± 0.001 | 4.071 ± 1.791 | 0.077 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.551 ± 0.013 | 0.992 ± 0.011 | 0.943 ± 0.071 | 1.791 ± 0.127 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.111 ± 0.001 | 0.098 ± 0.000 | 0.096 ± 0.005 | 0.043 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.289 ± 0.007 | 0.417 ± 0.004 | 0.358 ± 0.037 | 0.838 ± 0.040 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.175 ± 0.002 | 0.240 ± 0.003 | 0.148 ± 0.008 | 0.134 ± 0.002 | - | - |
| large | `eigh` | f64 | 4 | `1024x1024` | - | 111.645 ± 1.044 | 112.029 ± 0.470 | 242.111 ± 16.079 | 258.319 ± 58.270 | - | - |
| large | `eigh` | f64 | 4 | `128x128` | - | 2.194 ± 0.011 | 2.221 ± 0.014 | 7.546 ± 64.138 | 9.166 ± 2.447 | - | - |
| large | `eigh` | f64 | 4 | `256x256` | - | 9.464 ± 0.021 | 9.473 ± 0.037 | 12.021 ± 128.206 | 132.929 ± 118.075 | - | - |
| large | `eigh` | f64 | 4 | `512x512` | - | 23.424 ± 0.104 | 23.072 ± 0.042 | 44.090 ± 117.421 | 161.986 ± 20.664 | - | - |
| large | `eigh` | f64 | 4 | `64x64` | - | 0.442 ± 0.001 | 0.442 ± 0.004 | 0.289 ± 0.006 | 0.462 ± 0.006 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | - | 390.692 ± 2.732 | 212.474 ± 1.725 | 205.858 ± 3.449 | 266.314 ± 15.836 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 30.068 ± 0.138 | 15.225 ± 0.042 | 6.117 ± 0.333 | 11.278 ± 0.404 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 77.041 ± 0.387 | 45.155 ± 0.152 | 33.165 ± 0.170 | 52.541 ± 6.444 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | - | 263.089 ± 0.879 | 212.337 ± 1.185 | 139.981 ± 0.787 | 257.894 ± 8.933 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 20.050 ± 0.049 | 15.260 ± 0.069 | 5.259 ± 0.056 | 10.561 ± 0.918 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 52.730 ± 0.353 | 45.190 ± 0.396 | 25.870 ± 0.309 | 50.038 ± 3.555 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | - | 236.071 ± 1.844 | 104.080 ± 0.440 | 237.293 ± 2.212 | 195.873 ± 20.014 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 6.506 ± 0.072 | 2.921 ± 0.037 | 5.634 ± 0.101 | 5.896 ± 1.358 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 32.763 ± 0.233 | 15.894 ± 0.093 | 42.166 ± 0.833 | 30.253 ± 3.541 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | - | 216.169 ± 0.657 | 107.250 ± 0.724 | 119.023 ± 1.986 | 176.915 ± 6.844 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 5.103 ± 0.117 | 3.016 ± 0.050 | 2.702 ± 0.027 | 5.129 ± 3.361 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 30.289 ± 0.201 | 16.773 ± 0.347 | 18.294 ± 0.112 | 30.458 ± 7.328 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `1024x1024` | - | 19.170 ± 0.118 | 17.520 ± 0.030 | 27.194 ± 0.151 | 20.415 ± 0.145 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `128x128` | - | 0.083 ± 0.002 | 0.056 ± 0.000 | 0.093 ± 0.004 | 0.145 ± 0.017 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `256x256` | - | 0.395 ± 0.002 | 0.308 ± 0.002 | 0.453 ± 0.006 | 0.623 ± 0.070 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `512x512` | - | 2.505 ± 0.004 | 2.205 ± 0.006 | 5.200 ± 0.022 | 3.103 ± 0.516 | - | - |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | - | 0.032 ± 0.001 | 0.019 ± 0.000 | 0.028 ± 0.002 | 0.036 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `1024x1024` | - | 58.718 ± 0.369 | 52.536 ± 0.166 | 76.429 ± 0.266 | 60.738 ± 2.780 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `128x128` | - | 0.290 ± 0.008 | 0.158 ± 0.000 | 0.291 ± 0.004 | 0.297 ± 0.016 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `256x256` | - | 1.169 ± 0.005 | 0.919 ± 0.002 | 1.320 ± 0.106 | 1.425 ± 0.129 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `512x512` | - | 7.714 ± 0.064 | 6.706 ± 0.008 | 12.293 ± 0.078 | 8.566 ± 0.709 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | - | 0.128 ± 0.004 | 0.054 ± 0.000 | 0.107 ± 0.002 | 0.083 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | - | 328.310 ± 3.801 | 159.664 ± 1.490 | 165.909 ± 1.959 | 274.140 ± 28.374 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 9.429 ± 0.058 | 4.265 ± 0.036 | 4.273 ± 0.120 | 8.967 ± 0.885 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 48.576 ± 0.343 | 24.458 ± 0.389 | 28.271 ± 0.171 | 43.726 ± 4.475 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | - | 280.293 ± 0.550 | 159.462 ± 0.658 | 166.959 ± 2.489 | 247.898 ± 10.270 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 6.585 ± 0.304 | 4.261 ± 0.014 | 4.575 ± 0.104 | 9.064 ± 0.973 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 39.576 ± 0.912 | 24.310 ± 0.335 | 32.662 ± 0.638 | 46.715 ± 7.096 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `1024x1024,rhs=1` | - | 11.442 ± 0.086 | 30.105 ± 0.337 | 137.692 ± 88.692 | 44.587 ± 111.996 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `128x128,rhs=1` | - | 0.252 ± 0.024 | 0.380 ± 0.005 | 0.273 ± 0.007 | 0.221 ± 0.007 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `256x256,rhs=1` | - | 0.705 ± 0.014 | 1.273 ± 0.004 | 27.908 ± 1.394 | 2.182 ± 1.582 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `512x512,rhs=1` | - | 2.040 ± 0.038 | 5.161 ± 0.032 | 6.860 ± 20.900 | 133.780 ± 51.212 | - | - |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | - | 0.135 ± 0.010 | 0.138 ± 0.002 | 0.121 ± 0.004 | 0.069 ± 0.002 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | - | 31.496 ± 0.208 | 20.322 ± 0.151 | 12.965 ± 0.125 | 20.391 ± 0.324 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.499 ± 0.029 | 0.839 ± 0.015 | 1.501 ± 0.054 | 0.750 ± 0.017 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.964 ± 0.039 | 3.417 ± 0.073 | 2.894 ± 0.273 | 4.224 ± 0.785 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | - | 20.817 ± 0.306 | 19.380 ± 0.254 | 22.690 ± 0.625 | 19.405 ± 0.396 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.982 ± 0.049 | 0.831 ± 0.014 | 0.891 ± 0.028 | 0.787 ± 0.008 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 3.818 ± 0.157 | 3.451 ± 0.036 | 4.340 ± 0.086 | 4.291 ± 0.068 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `1024x1024` | - | 299.819 ± 3.875 | 430.649 ± 1.713 | 392.520 ± 8.630 | 526.553 ± 10.150 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `128x128` | - | 3.527 ± 0.044 | 4.888 ± 0.009 | 3.310 ± 0.039 | 4.181 ± 0.033 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `256x256` | - | 19.001 ± 0.183 | 27.392 ± 1.159 | 15.329 ± 0.060 | 18.125 ± 0.567 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `512x512` | - | 50.487 ± 0.210 | 78.344 ± 0.690 | 92.374 ± 0.584 | 111.166 ± 4.882 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | - | 0.959 ± 0.009 | 1.221 ± 0.003 | 0.838 ± 0.028 | 0.836 ± 0.051 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | - | 835.165 ± 4.299 | 408.223 ± 5.566 | 434.952 ± 6.654 | 528.512 ± 9.372 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 55.795 ± 0.339 | 27.317 ± 0.251 | 16.671 ± 0.246 | 19.061 ± 0.568 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 140.747 ± 0.946 | 77.956 ± 0.178 | 89.656 ± 1.710 | 109.057 ± 3.049 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | - | 557.170 ± 8.789 | 431.547 ± 3.515 | 353.103 ± 1.849 | 545.535 ± 18.525 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 36.898 ± 0.056 | 27.297 ± 0.079 | 15.406 ± 0.149 | 19.041 ± 0.279 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 95.165 ± 0.453 | 77.926 ± 0.502 | 84.260 ± 2.054 | 110.073 ± 2.796 | - | - |
| large | `matmul` | f64 | 4 | `1024x1024` | - | 17.460 ± 0.097 | 17.439 ± 0.017 | 22.391 ± 0.140 | 19.580 ± 0.837 | - | - |
| large | `matmul` | f64 | 4 | `128x128` | - | 0.051 ± 0.000 | 0.051 ± 0.000 | 0.100 ± 0.003 | 0.119 ± 0.010 | - | - |
| large | `matmul` | f64 | 4 | `256x256` | - | 0.298 ± 0.001 | 0.296 ± 0.001 | 0.467 ± 0.007 | 0.607 ± 0.068 | - | - |
| large | `matmul` | f64 | 4 | `512x512` | - | 2.191 ± 0.007 | 2.192 ± 0.004 | 3.938 ± 0.016 | 3.207 ± 0.689 | - | - |
| large | `matmul_rect` | f64 | 4 | `1024x256 * 256x1024` | - | 4.344 ± 0.019 | 4.336 ± 0.006 | 5.860 ± 0.057 | 5.463 ± 0.068 | - | - |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | - | 1.166 ± 0.003 | 1.165 ± 0.005 | 1.583 ± 0.025 | 1.617 ± 0.070 | - | - |
| large | `qr` | f64 | 4 | `1024x1024` | - | 42.850 ± 0.368 | 42.933 ± 0.406 | 67.028 ± 0.725 | 86.933 ± 1.450 | - | - |
| large | `qr` | f64 | 4 | `128x128` | - | 0.557 ± 0.004 | 0.553 ± 0.004 | 0.510 ± 0.021 | 0.853 ± 0.016 | - | - |
| large | `qr` | f64 | 4 | `256x256` | - | 1.382 ± 0.011 | 1.378 ± 0.016 | 2.184 ± 0.031 | 3.243 ± 0.155 | - | - |
| large | `qr` | f64 | 4 | `512x512` | - | 7.112 ± 0.040 | 7.139 ± 0.072 | 12.900 ± 0.051 | 17.928 ± 0.541 | - | - |
| large | `qr` | f64 | 4 | `64x64` | - | 0.134 ± 0.001 | 0.134 ± 0.002 | 0.144 ± 0.004 | 0.119 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=1` | - | 9.091 ± 0.024 | 9.039 ± 0.107 | 46.694 ± 115.212 | 138.536 ± 115.299 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=16` | - | 9.764 ± 0.064 | 9.594 ± 0.036 | 134.281 ± 84.950 | 123.461 ± 113.721 | - | - |
| large | `solve` | f64 | 4 | `1024x1024,rhs=64` | - | 11.723 ± 0.078 | 11.397 ± 0.068 | 61.161 ± 63.146 | 66.002 ± 80.618 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=1` | - | 0.112 ± 0.000 | 0.105 ± 0.001 | 0.185 ± 0.004 | 0.185 ± 0.004 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=16` | - | 0.141 ± 0.002 | 0.124 ± 0.002 | 8.771 ± 2.958 | 5.515 ± 7.661 | - | - |
| large | `solve` | f64 | 4 | `128x128,rhs=64` | - | 0.225 ± 0.004 | 0.174 ± 0.004 | 8.310 ± 0.182 | 3.289 ± 3.711 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=1` | - | 0.422 ± 0.006 | 0.379 ± 0.003 | 21.948 ± 2.002 | 1.618 ± 66.049 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=16` | - | 0.500 ± 0.006 | 0.436 ± 0.003 | 32.673 ± 24.681 | 4.808 ± 83.672 | - | - |
| large | `solve` | f64 | 4 | `256x256,rhs=64` | - | 0.737 ± 0.006 | 0.598 ± 0.005 | 28.917 ± 2.674 | 2.914 ± 1.126 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=1` | - | 1.792 ± 0.019 | 1.609 ± 0.014 | 2.104 ± 0.029 | 127.544 ± 122.472 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=16` | - | 2.035 ± 0.023 | 1.808 ± 0.009 | 33.276 ± 109.093 | 24.954 ± 116.827 | - | - |
| large | `solve` | f64 | 4 | `512x512,rhs=64` | - | 2.745 ± 0.013 | 2.443 ± 0.009 | 4.022 ± 27.714 | 107.403 ± 110.620 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | - | 0.035 ± 0.000 | 0.035 ± 0.000 | 0.054 ± 0.003 | 0.051 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | - | 0.045 ± 0.001 | 0.038 ± 0.001 | 0.067 ± 0.003 | 0.101 ± 0.002 | - | - |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | - | 0.083 ± 0.001 | 0.057 ± 0.001 | 0.085 ± 0.003 | 0.143 ± 0.008 | - | - |
| large | `svd` | f64 | 4 | `1024x1024` | - | 256.586 ± 0.723 | 241.085 ± 1.675 | 330.199 ± 6.824 | 413.941 ± 3.079 | - | - |
| large | `svd` | f64 | 4 | `128x128` | - | 3.287 ± 0.019 | 3.302 ± 0.038 | 2.943 ± 0.028 | 3.332 ± 0.024 | - | - |
| large | `svd` | f64 | 4 | `256x256` | - | 17.981 ± 0.086 | 18.022 ± 0.041 | 14.199 ± 0.106 | 15.627 ± 0.379 | - | - |
| large | `svd` | f64 | 4 | `512x512` | - | 44.903 ± 0.359 | 44.990 ± 0.582 | 64.101 ± 1.616 | 91.629 ± 0.654 | - | - |
| large | `svd` | f64 | 4 | `64x64` | - | 0.827 ± 0.002 | 0.826 ± 0.005 | 0.724 ± 0.018 | 0.768 ± 0.020 | - | - |
| small | `eigh` | f64 | 4 | `16x16` | - | 0.037 ± 0.000 | 0.035 ± 0.000 | 0.033 ± 0.000 | 0.045 ± 0.001 | - | - |
| small | `eigh` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `32x32` | - | 0.121 ± 0.000 | 0.120 ± 0.000 | 0.111 ± 0.001 | 0.126 ± 0.003 | - | - |
| small | `eigh` | f64 | 4 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 4 | `8x8` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.152 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.145 ± 0.001 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `32x32` | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.154 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.148 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.152 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `16x16` | - | 0.355 ± 0.007 | 0.088 ± 0.000 | 0.159 ± 0.002 | 0.043 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.249 ± 0.007 | 0.034 ± 0.000 | 0.121 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `32x32` | - | 0.642 ± 0.012 | 0.243 ± 0.002 | 0.261 ± 0.007 | 0.131 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.258 ± 0.003 | 0.038 ± 0.000 | 0.126 ± 0.007 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.282 ± 0.003 | 0.049 ± 0.001 | 0.135 ± 0.006 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `16x16` | - | 0.143 ± 0.001 | 0.086 ± 0.001 | 0.135 ± 0.003 | 0.043 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.079 ± 0.001 | 0.032 ± 0.000 | 0.106 ± 0.010 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `32x32` | - | 0.328 ± 0.004 | 0.241 ± 0.001 | 0.228 ± 0.008 | 0.132 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.084 ± 0.002 | 0.036 ± 0.000 | 0.110 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.097 ± 0.001 | 0.048 ± 0.000 | 0.116 ± 0.004 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `16x16` | - | 0.964 ± 0.016 | 0.053 ± 0.000 | 0.228 ± 0.004 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.928 ± 0.010 | 0.044 ± 0.000 | 0.208 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `32x32` | - | 1.028 ± 0.021 | 0.080 ± 0.000 | 0.285 ± 0.005 | 0.059 ± 0.028 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.948 ± 0.047 | 0.044 ± 0.000 | 0.203 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.954 ± 0.014 | 0.046 ± 0.000 | 0.212 ± 0.010 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `16x16` | - | 0.216 ± 0.001 | 0.050 ± 0.001 | 0.179 ± 0.008 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.196 ± 0.002 | 0.039 ± 0.000 | 0.164 ± 0.009 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `32x32` | - | 0.265 ± 0.005 | 0.078 ± 0.001 | 0.220 ± 0.062 | 0.060 ± 0.051 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.199 ± 0.005 | 0.040 ± 0.000 | 0.162 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.272 ± 0.063 | 0.042 ± 0.000 | 0.166 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `16x16` | - | 0.065 ± 0.001 | 0.017 ± 0.000 | 0.047 ± 0.002 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | - | 0.061 ± 0.001 | 0.016 ± 0.000 | 0.042 ± 0.001 | 0.013 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `32x32` | - | 0.075 ± 0.002 | 0.024 ± 0.000 | 0.063 ± 0.002 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | - | 0.061 ± 0.001 | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | - | 0.062 ± 0.001 | 0.016 ± 0.000 | 0.044 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `16x16` | - | 1.367 ± 0.016 | 0.078 ± 0.000 | 0.160 ± 0.006 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 1.277 ± 0.042 | 0.055 ± 0.001 | 0.144 ± 0.020 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `32x32` | - | 1.442 ± 0.008 | 0.143 ± 0.001 | 0.208 ± 0.007 | 0.122 ± 0.053 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 1.301 ± 0.019 | 0.056 ± 0.001 | 0.143 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 1.322 ± 0.027 | 0.061 ± 0.000 | 0.148 ± 0.003 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `16x16` | - | 0.205 ± 0.001 | 0.077 ± 0.001 | 0.168 ± 0.007 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.175 ± 0.003 | 0.054 ± 0.000 | 0.146 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `32x32` | - | 0.293 ± 0.004 | 0.141 ± 0.001 | 0.207 ± 0.005 | 0.063 ± 0.017 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.179 ± 0.004 | 0.055 ± 0.000 | 0.149 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.183 ± 0.004 | 0.059 ± 0.001 | 0.155 ± 0.008 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `16x16,rhs=1` | - | 0.086 ± 0.001 | 0.057 ± 0.001 | 0.070 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | - | 0.082 ± 0.001 | 0.049 ± 0.000 | 0.065 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `32x32,rhs=1` | - | 0.096 ± 0.002 | 0.073 ± 0.001 | 0.082 ± 0.001 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | - | 0.082 ± 0.001 | 0.048 ± 0.000 | 0.067 ± 0.001 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | - | 0.085 ± 0.000 | 0.051 ± 0.000 | 0.068 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `16x16,rhs=1` | - | 0.194 ± 0.003 | 0.039 ± 0.001 | 0.340 ± 0.010 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.185 ± 0.004 | 0.034 ± 0.000 | 0.332 ± 0.016 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `32x32,rhs=1` | - | 0.215 ± 0.003 | 0.050 ± 0.000 | 0.355 ± 0.009 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.186 ± 0.004 | 0.034 ± 0.000 | 0.334 ± 0.008 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.189 ± 0.003 | 0.035 ± 0.000 | 0.330 ± 0.008 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `16x16,rhs=1` | - | 0.098 ± 0.001 | 0.038 ± 0.000 | 0.145 ± 0.003 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.091 ± 0.002 | 0.033 ± 0.000 | 0.140 ± 0.008 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `32x32,rhs=1` | - | 0.110 ± 0.002 | 0.049 ± 0.000 | 0.165 ± 0.006 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.093 ± 0.002 | 0.033 ± 0.000 | 0.138 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.094 ± 0.002 | 0.035 ± 0.000 | 0.141 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `16x16` | - | 0.127 ± 0.005 | 0.100 ± 0.001 | 0.109 ± 0.001 | 0.071 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | - | 0.068 ± 0.001 | 0.023 ± 0.000 | 0.047 ± 0.001 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `32x32` | - | 0.287 ± 0.007 | 0.314 ± 0.001 | 0.268 ± 0.002 | 0.209 ± 0.003 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | - | 0.072 ± 0.002 | 0.028 ± 0.000 | 0.055 ± 0.002 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | - | 0.084 ± 0.001 | 0.043 ± 0.000 | 0.068 ± 0.002 | 0.023 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `16x16` | - | 0.396 ± 0.005 | 0.101 ± 0.000 | 0.265 ± 0.016 | 0.069 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.216 ± 0.005 | 0.024 ± 0.000 | 0.185 ± 0.005 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `32x32` | - | 0.876 ± 0.012 | 0.313 ± 0.001 | 0.451 ± 0.012 | 0.203 ± 0.004 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.226 ± 0.002 | 0.030 ± 0.000 | 0.195 ± 0.009 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.262 ± 0.005 | 0.045 ± 0.000 | 0.215 ± 0.010 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `16x16` | - | 0.193 ± 0.001 | 0.100 ± 0.001 | 0.179 ± 0.005 | 0.069 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.080 ± 0.004 | 0.023 ± 0.000 | 0.113 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `32x32` | - | 0.505 ± 0.003 | 0.313 ± 0.001 | 0.349 ± 0.013 | 0.205 ± 0.002 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.089 ± 0.002 | 0.029 ± 0.000 | 0.120 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.001 | 0.044 ± 0.000 | 0.136 ± 0.003 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `16x16` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `32x32` | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `matmul` | f64 | 4 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.002 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 4 | `64x64` | - | 0.016 ± 0.001 | 0.017 ± 0.000 | 0.028 ± 0.000 | 0.036 ± 0.005 | - | - |
| small | `matmul` | f64 | 4 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `16x16` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `qr` | f64 | 4 | `32x32` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.032 ± 0.001 | 0.036 ± 0.003 | - | - |
| small | `qr` | f64 | 4 | `4x4` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 4 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=1` | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `16x16,rhs=4` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=1` | - | 0.015 ± 0.000 | 0.016 ± 0.000 | 0.029 ± 0.002 | 0.015 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `32x32,rhs=4` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `16x16` | - | 0.062 ± 0.007 | 0.061 ± 0.000 | 0.065 ± 0.001 | 0.072 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `2x2` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 4 | `32x32` | - | 0.212 ± 0.001 | 0.212 ± 0.003 | 0.218 ± 0.001 | 0.204 ± 0.006 | - | - |
| small | `svd` | f64 | 4 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.000 | - | - |
| small | `svd` | f64 | 4 | `8x8` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.000 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 13.9x faster than the slowest successful cell (`jax-cpu`, 1.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 1.587 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 21.9x faster than the slowest successful cell (`jax-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 21.2x faster than the slowest successful cell (`jax-cpu`, 0.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 24.6x faster than the slowest successful cell (`jax-cpu`, 0.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 25.8x faster than the slowest successful cell (`jax-cpu`, 0.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 13.6x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.157 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 20.4x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `pytorch-cpu` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 0.453 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 18.5x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-eager` is 12.2x faster than the slowest successful cell (`jax-cpu`, 1.881 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_solve` (f64, threads=4, shape=`16x16xbatch256 (native batch layout),rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 1.881 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 10.505 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.263 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 22.2x faster than the slowest successful cell (`pytorch-cpu`, 2.683 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.716 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 0.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 7.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 44.1x faster than the slowest successful cell (`pytorch-cpu`, 7.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 1.927 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 31.8x faster than the slowest successful cell (`pytorch-cpu`, 1.927 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=4, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 0.561 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `jax-cpu` is 53.2x faster than the slowest successful cell (`pytorch-cpu`, 4.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `tenferro-eager` is 32.9x faster than the slowest successful cell (`pytorch-cpu`, 4.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_solve_backward` (f64, threads=4, shape=`4x4xbatch64 (native batch layout),rhs=1`): `tenferro-trace` is 34.1x faster than the slowest successful cell (`pytorch-cpu`, 4.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`jax-cpu`, 132.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `tenferro-eager` is 14.0x faster than the slowest successful cell (`jax-cpu`, 132.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=4, shape=`256x256`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`jax-cpu`, 132.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`1024x1024,rhs=1`): `tenferro-eager` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 137.692 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 27.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-eager` is 39.6x faster than the slowest successful cell (`pytorch-cpu`, 27.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 27.908 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`jax-cpu`, 133.780 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-eager` is 65.6x faster than the slowest successful cell (`jax-cpu`, 133.780 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`jax-cpu`, 133.780 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=1`): `tenferro-eager` is 15.2x faster than the slowest successful cell (`jax-cpu`, 138.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=1`): `tenferro-trace` is 15.3x faster than the slowest successful cell (`jax-cpu`, 138.536 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=16`): `tenferro-eager` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 134.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`1024x1024,rhs=16`): `tenferro-trace` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 134.281 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-eager` is 62.2x faster than the slowest successful cell (`pytorch-cpu`, 8.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=16`): `tenferro-trace` is 70.8x faster than the slowest successful cell (`pytorch-cpu`, 8.771 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-eager` is 37.0x faster than the slowest successful cell (`pytorch-cpu`, 8.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`128x128,rhs=64`): `tenferro-trace` is 47.7x faster than the slowest successful cell (`pytorch-cpu`, 8.310 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `jax-cpu` is 13.6x faster than the slowest successful cell (`pytorch-cpu`, 21.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-eager` is 52.0x faster than the slowest successful cell (`pytorch-cpu`, 21.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=1`): `tenferro-trace` is 57.9x faster than the slowest successful cell (`pytorch-cpu`, 21.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-eager` is 65.3x faster than the slowest successful cell (`pytorch-cpu`, 32.673 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=16`): `tenferro-trace` is 74.9x faster than the slowest successful cell (`pytorch-cpu`, 32.673 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-eager` is 39.3x faster than the slowest successful cell (`pytorch-cpu`, 28.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`256x256,rhs=64`): `tenferro-trace` is 48.3x faster than the slowest successful cell (`pytorch-cpu`, 28.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `pytorch-cpu` is 60.6x faster than the slowest successful cell (`jax-cpu`, 127.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-eager` is 71.2x faster than the slowest successful cell (`jax-cpu`, 127.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=1`): `tenferro-trace` is 79.3x faster than the slowest successful cell (`jax-cpu`, 127.544 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-eager` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 33.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=16`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 33.276 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=64`): `pytorch-cpu` is 26.7x faster than the slowest successful cell (`jax-cpu`, 107.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=64`): `tenferro-eager` is 39.1x faster than the slowest successful cell (`jax-cpu`, 107.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=4, shape=`512x512,rhs=64`): `tenferro-trace` is 44.0x faster than the slowest successful cell (`jax-cpu`, 107.403 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-eager` is 28.4x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-eager` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-eager` is 19.8x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-eager` is 30.7x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-eager` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.152 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 0.249 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 0.258 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 17.8x faster than the slowest successful cell (`tenferro-eager`, 0.282 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 47.5x faster than the slowest successful cell (`tenferro-eager`, 0.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.964 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 66.9x faster than the slowest successful cell (`tenferro-eager`, 0.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 21.3x faster than the slowest successful cell (`tenferro-eager`, 0.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`32x32`): `jax-cpu` is 17.3x faster than the slowest successful cell (`tenferro-eager`, 1.028 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 1.028 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 63.5x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`tenferro-eager`, 0.948 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 44.4x faster than the slowest successful cell (`tenferro-eager`, 0.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`tenferro-eager`, 0.954 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.2x faster than the slowest successful cell (`tenferro-eager`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 14.2x faster than the slowest successful cell (`tenferro-eager`, 0.196 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-eager`, 0.199 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 13.4x faster than the slowest successful cell (`tenferro-eager`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 71.6x faster than the slowest successful cell (`tenferro-eager`, 1.367 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`16x16`): `tenferro-trace` is 17.4x faster than the slowest successful cell (`tenferro-eager`, 1.367 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 95.9x faster than the slowest successful cell (`tenferro-eager`, 1.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`2x2`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`tenferro-eager`, 1.277 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`32x32`): `jax-cpu` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 1.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`32x32`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 1.442 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 84.1x faster than the slowest successful cell (`tenferro-eager`, 1.301 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`4x4`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`tenferro-eager`, 1.301 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 66.0x faster than the slowest successful cell (`tenferro-eager`, 1.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=4, shape=`8x8`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 1.322 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`16x16`): `jax-cpu` is 11.4x faster than the slowest successful cell (`tenferro-eager`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 13.5x faster than the slowest successful cell (`tenferro-eager`, 0.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.179 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 0.183 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`16x16,rhs=1`): `jax-cpu` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.340 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`2x2,rhs=1`): `jax-cpu` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 0.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`32x32,rhs=1`): `jax-cpu` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.355 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`4x4,rhs=1`): `jax-cpu` is 24.3x faster than the slowest successful cell (`pytorch-cpu`, 0.334 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=4, shape=`8x8,rhs=1`): `jax-cpu` is 22.3x faster than the slowest successful cell (`pytorch-cpu`, 0.330 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`2x2`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 0.216 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-eager`, 0.226 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=4, shape=`8x8`): `jax-cpu` is 12.1x faster than the slowest successful cell (`tenferro-eager`, 0.262 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

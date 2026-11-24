# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

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

- OMP_NUM_THREADS: `8`
- OMP_THREAD_LIMIT: `8`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `8`
- OPENBLAS_NUM_THREADS: `8`
- GOTO_NUM_THREADS: `8`
- MKL_NUM_THREADS: `8`
- VECLIB_MAXIMUM_THREADS: `8`
- VECLIB_NUM_THREADS: `8`
- NUMEXPR_NUM_THREADS: `8`
- BLIS_NUM_THREADS: `8`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=8`

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 8

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_135202/cpu_ops_t8_20261124_135202.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 4.184 ± 0.013 | 4.148 ± 0.008 | 27.206 ± 0.089 | 10.380 ± 2.947 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.073 ± 0.001 | 0.069 ± 0.000 | 0.409 ± 0.002 | 0.391 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 1.052 ± 0.003 | 1.032 ± 0.001 | 6.868 ± 0.029 | 2.537 ± 0.414 | - | - |
| batched | `batched_eigh` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.264 ± 0.001 | 0.263 ± 0.001 | 1.684 ± 0.011 | 0.570 ± 0.017 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.543 ± 0.001 | 0.544 ± 0.002 | 0.545 ± 0.005 | 0.425 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.141 ± 0.000 | 0.140 ± 0.000 | 0.143 ± 0.001 | 0.110 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.000 | 0.039 ± 0.000 | 0.042 ± 0.000 | 0.032 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.367 ± 0.002 | 0.366 ± 0.002 | 2.701 ± 0.004 | 1.706 ± 0.031 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.048 ± 0.000 | 0.050 ± 0.000 | 0.052 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.717 ± 0.002 | 0.717 ± 0.003 | 0.678 ± 0.005 | 0.663 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.185 ± 0.001 | 0.182 ± 0.001 | 0.176 ± 0.001 | 0.172 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 1.121 ± 0.011 | 1.113 ± 0.020 | 7.968 ± 0.031 | 3.607 ± 0.059 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.141 ± 0.001 | 0.139 ± 0.000 | 0.132 ± 0.001 | 0.146 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.284 ± 0.001 | 0.282 ± 0.000 | 2.001 ± 0.006 | 0.864 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.077 ± 0.000 | 0.075 ± 0.000 | 0.504 ± 0.002 | 0.562 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 0.060 ± 0.009 | 0.060 ± 0.000 | 0.247 ± 0.007 | 1.595 ± 0.039 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.023 ± 0.001 | 0.158 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.020 ± 0.000 | 0.020 ± 0.000 | 0.081 ± 0.006 | 0.336 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.036 ± 0.002 | 0.170 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.149 ± 0.002 | 0.148 ± 0.000 | 0.044 ± 0.002 | 0.159 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.008 ± 0.001 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.149 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.041 ± 0.000 | 0.041 ± 0.000 | 0.022 ± 0.001 | 0.156 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.014 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.001 | 0.151 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.027 ± 0.001 | 0.026 ± 0.000 | 0.071 ± 0.001 | 0.280 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.001 | 0.155 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.048 ± 0.001 | 0.044 ± 0.000 | 0.053 ± 0.002 | 0.162 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.001 | 0.153 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.029 ± 0.001 | 0.029 ± 0.000 | 0.076 ± 0.006 | 0.470 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | 0.155 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.033 ± 0.001 | 0.171 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.022 ± 0.002 | 0.156 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 1.239 ± 0.017 | 1.232 ± 0.002 | 5.639 ± 0.036 | 2.663 ± 0.069 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.029 ± 0.000 | 0.026 ± 0.000 | 0.099 ± 0.003 | 0.135 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.314 ± 0.001 | 0.316 ± 0.001 | 1.432 ± 0.009 | 0.583 ± 0.043 | - | - |
| batched | `batched_qr` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.086 ± 0.001 | 0.084 ± 0.000 | 0.373 ± 0.007 | 0.373 ± 0.007 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.319 ± 0.000 | 0.318 ± 0.001 | 0.224 ± 0.001 | 0.329 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.085 ± 0.000 | 0.084 ± 0.000 | 0.062 ± 0.001 | 0.087 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.025 ± 0.000 | 0.021 ± 0.001 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.115 ± 0.001 | 0.111 ± 0.003 | 0.652 ± 0.006 | 0.961 ± 0.021 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.001 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.209 ± 0.001 | 0.206 ± 0.001 | 0.167 ± 0.001 | 0.246 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.056 ± 0.000 | 0.048 ± 0.000 | 0.065 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.342 ± 0.006 | 0.332 ± 0.002 | 1.656 ± 0.011 | 1.011 ± 0.013 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.046 ± 0.000 | 0.045 ± 0.000 | 0.034 ± 0.001 | 0.048 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.090 ± 0.000 | 0.090 ± 0.000 | 0.427 ± 0.003 | 0.688 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.167 ± 0.001 | 0.163 ± 0.000 | 0.114 ± 0.001 | 0.175 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 0.312 ± 0.004 | 0.392 ± 0.004 | 0.774 ± 0.051 | 2.054 ± 0.078 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.046 ± 0.000 | 0.044 ± 0.000 | 0.036 ± 0.003 | 0.062 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.085 ± 0.001 | 0.111 ± 0.001 | 0.223 ± 0.017 | 0.531 ± 0.023 | - | - |
| batched | `batched_solve` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.029 ± 0.000 | 0.039 ± 0.000 | 0.079 ± 0.005 | 0.205 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.214 ± 0.001 | 0.166 ± 0.000 | 0.083 ± 0.000 | 0.360 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.020 ± 0.000 | 0.017 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.060 ± 0.001 | 0.049 ± 0.000 | 0.038 ± 0.001 | 0.095 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.025 ± 0.001 | 0.029 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.072 ± 0.026 | 0.072 ± 0.001 | 0.119 ± 0.001 | 0.512 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.022 ± 0.000 | 0.020 ± 0.000 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.085 ± 0.000 | 0.076 ± 0.000 | 0.045 ± 0.001 | 0.135 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.028 ± 0.000 | 0.027 ± 0.000 | 0.027 ± 0.001 | 0.041 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.118 ± 0.002 | 0.148 ± 0.001 | 0.319 ± 0.009 | 0.962 ± 0.020 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.001 | 0.022 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.037 ± 0.001 | 0.048 ± 0.001 | 0.104 ± 0.004 | 0.264 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.000 | 0.057 ± 0.000 | 0.042 ± 0.002 | 0.075 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 7.281 ± 0.019 | 7.269 ± 0.019 | 60.338 ± 0.094 | 15.434 ± 0.064 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.121 ± 0.001 | 0.120 ± 0.000 | 0.922 ± 0.002 | 0.382 ± 0.009 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 1.814 ± 0.004 | 1.814 ± 0.003 | 15.029 ± 0.061 | 3.926 ± 0.760 | - | - |
| batched | `batched_svd` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.460 ± 0.002 | 0.459 ± 0.002 | 3.758 ± 0.022 | 1.045 ± 0.011 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 1.299 ± 0.003 | 1.305 ± 0.002 | 1.380 ± 0.005 | 1.542 ± 0.013 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.028 ± 0.000 | 0.026 ± 0.000 | 0.031 ± 0.000 | 0.037 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.331 ± 0.001 | 0.330 ± 0.000 | 0.354 ± 0.001 | 0.392 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.000 | 0.087 ± 0.000 | 0.096 ± 0.000 | 0.109 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.604 ± 0.001 | 0.601 ± 0.004 | 5.598 ± 0.015 | 1.648 ± 0.006 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.079 ± 0.000 | 0.078 ± 0.000 | 0.095 ± 0.000 | 0.092 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 1.176 ± 0.003 | 1.181 ± 0.007 | 1.396 ± 0.007 | 1.280 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.001 | 0.299 ± 0.000 | 0.357 ± 0.001 | 0.322 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 1.919 ± 0.007 | 1.900 ± 0.005 | 18.788 ± 0.084 | 4.166 ± 1.004 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.241 ± 0.001 | 0.239 ± 0.000 | 0.301 ± 0.001 | 0.259 ± 0.002 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.482 ± 0.001 | 0.480 ± 0.002 | 4.716 ± 0.016 | 0.895 ± 0.085 | - | - |
| batched | `batched_svd` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.128 ± 0.002 | 0.126 ± 0.000 | 1.179 ± 0.008 | 0.702 ± 0.017 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout)` | - | 1.275 ± 0.067 | 0.228 ± 0.003 | 9.412 ± 0.066 | 3.483 ± 0.084 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch16 (native batch layout)` | - | 0.098 ± 0.002 | 0.025 ± 0.000 | 0.259 ± 0.009 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch256 (native batch layout)` | - | 0.301 ± 0.010 | 0.077 ± 0.000 | 2.580 ± 0.030 | 0.821 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `16x16xbatch64 (native batch layout)` | - | 0.166 ± 0.007 | 0.037 ± 0.000 | 0.722 ± 0.037 | 0.204 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout)` | - | 0.535 ± 0.004 | 0.473 ± 0.001 | 0.196 ± 0.001 | 0.027 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch16 (native batch layout)` | - | 0.071 ± 0.001 | 0.023 ± 0.000 | 0.106 ± 0.001 | 0.014 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch256 (native batch layout)` | - | 0.180 ± 0.002 | 0.130 ± 0.000 | 0.126 ± 0.001 | 0.023 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `2x2xbatch64 (native batch layout)` | - | 0.092 ± 0.001 | 0.045 ± 0.000 | 0.109 ± 0.000 | 0.016 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout)` | - | 0.241 ± 0.003 | 0.089 ± 0.000 | 0.309 ± 0.033 | 0.811 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch16 (native batch layout)` | - | 0.072 ± 0.001 | 0.024 ± 0.000 | 0.111 ± 0.002 | 0.019 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch256 (native batch layout)` | - | 0.204 ± 0.005 | 0.138 ± 0.001 | 0.219 ± 0.002 | 0.210 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `4x4xbatch64 (native batch layout)` | - | 0.098 ± 0.002 | 0.047 ± 0.000 | 0.132 ± 0.003 | 0.058 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout)` | - | 0.326 ± 0.008 | 0.106 ± 0.001 | 7.132 ± 0.046 | 1.330 ± 0.015 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch16 (native batch layout)` | - | 0.078 ± 0.000 | 0.026 ± 0.000 | 0.220 ± 0.002 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch256 (native batch layout)` | - | 0.170 ± 0.003 | 0.044 ± 0.000 | 1.840 ± 0.007 | 0.334 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 8 | `8x8xbatch64 (native batch layout)` | - | 0.102 ± 0.002 | 0.030 ± 0.000 | 0.551 ± 0.002 | 0.090 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 1.035 ± 0.034 | 1.649 ± 0.009 | 1.743 ± 0.024 | 4.132 ± 0.079 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.150 ± 0.004 | 0.180 ± 0.001 | 0.127 ± 0.007 | 0.097 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.317 ± 0.008 | 0.475 ± 0.002 | 0.566 ± 0.055 | 1.033 ± 0.024 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.203 ± 0.005 | 0.259 ± 0.001 | 0.267 ± 0.010 | 0.351 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.470 ± 0.006 | 0.789 ± 0.003 | 0.213 ± 0.006 | 0.720 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.096 ± 0.002 | 0.063 ± 0.000 | 0.076 ± 0.000 | 0.029 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.184 ± 0.002 | 0.237 ± 0.001 | 0.110 ± 0.006 | 0.189 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.113 ± 0.002 | 0.097 ± 0.000 | 0.085 ± 0.002 | 0.056 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.430 ± 0.007 | 0.555 ± 0.004 | 0.341 ± 0.014 | 1.033 ± 0.041 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.097 ± 0.003 | 0.068 ± 0.000 | 0.080 ± 0.001 | 0.025 ± 0.000 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.222 ± 0.002 | 0.327 ± 0.001 | 0.142 ± 0.005 | 0.269 ± 0.004 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.124 ± 0.003 | 0.120 ± 0.000 | 0.093 ± 0.001 | 0.079 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.519 ± 0.123 | 0.654 ± 0.006 | 0.867 ± 0.019 | 1.791 ± 0.094 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.112 ± 0.004 | 0.100 ± 0.000 | 0.094 ± 0.001 | 0.042 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.266 ± 0.005 | 0.355 ± 0.001 | 0.341 ± 0.010 | 0.478 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 8 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.177 ± 0.005 | 0.239 ± 0.001 | 0.142 ± 0.002 | 0.134 ± 0.003 | - | - |
| large | `eigh` | f64 | 8 | `1024x1024` | - | 88.502 ± 0.189 | 88.470 ± 0.248 | 201.839 ± 15.388 | 257.339 ± 87.175 | - | - |
| large | `eigh` | f64 | 8 | `128x128` | - | 2.210 ± 0.003 | 2.231 ± 0.007 | 3.858 ± 0.034 | 3.593 ± 51.537 | - | - |
| large | `eigh` | f64 | 8 | `256x256` | - | 9.437 ± 0.027 | 9.468 ± 0.025 | 150.789 ± 36.428 | 18.393 ± 109.129 | - | - |
| large | `eigh` | f64 | 8 | `512x512` | - | 21.352 ± 0.061 | 21.232 ± 0.108 | 159.341 ± 15.005 | 155.125 ± 51.831 | - | - |
| large | `eigh` | f64 | 8 | `64x64` | - | 0.448 ± 0.001 | 0.444 ± 0.002 | 0.281 ± 0.003 | 0.466 ± 0.004 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `1024x1024` | - | 288.497 ± 1.088 | 172.476 ± 0.369 | 132.890 ± 1.029 | 209.028 ± 19.220 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `256x256` | - | 30.314 ± 0.178 | 15.152 ± 0.068 | 4.797 ± 0.128 | 11.566 ± 1.861 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 8 | `512x512` | - | 67.781 ± 0.410 | 41.459 ± 0.076 | 22.913 ± 0.881 | 47.599 ± 8.322 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `1024x1024` | - | 200.425 ± 1.229 | 172.145 ± 0.112 | 95.517 ± 0.671 | 199.308 ± 8.888 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `256x256` | - | 20.128 ± 0.079 | 15.178 ± 0.023 | 4.335 ± 0.132 | 11.578 ± 1.745 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 8 | `512x512` | - | 45.862 ± 0.113 | 41.390 ± 0.170 | 17.907 ± 0.093 | 45.443 ± 4.763 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `1024x1024` | - | 153.869 ± 0.765 | 64.606 ± 0.295 | 148.207 ± 1.491 | 125.824 ± 17.963 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `256x256` | - | 5.878 ± 0.068 | 2.451 ± 0.012 | 4.054 ± 0.057 | 6.156 ± 2.999 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 8 | `512x512` | - | 22.626 ± 0.121 | 11.172 ± 0.019 | 27.153 ± 0.287 | 26.547 ± 6.721 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `1024x1024` | - | 127.946 ± 2.005 | 66.608 ± 0.375 | 71.698 ± 0.508 | 128.366 ± 10.394 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `256x256` | - | 4.205 ± 0.076 | 2.526 ± 0.006 | 2.082 ± 0.016 | 5.405 ± 1.228 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 8 | `512x512` | - | 20.203 ± 0.086 | 11.638 ± 0.012 | 11.469 ± 0.141 | 23.060 ± 8.901 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `1024x1024` | - | 10.702 ± 0.038 | 9.207 ± 0.032 | 15.038 ± 0.065 | 12.090 ± 1.107 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `128x128` | - | 0.072 ± 0.002 | 0.044 ± 0.000 | 0.062 ± 0.003 | 0.106 ± 0.003 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `256x256` | - | 0.276 ± 0.002 | 0.179 ± 0.002 | 0.276 ± 0.002 | 0.380 ± 0.016 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `512x512` | - | 1.504 ± 0.011 | 1.179 ± 0.005 | 2.909 ± 0.035 | 2.074 ± 0.060 | - | - |
| large | `grad_sum_matmul` | f64 | 8 | `64x64` | - | 0.030 ± 0.002 | 0.019 ± 0.000 | 0.023 ± 0.002 | 0.038 ± 0.001 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `1024x1024` | - | 32.839 ± 0.283 | 27.153 ± 0.019 | 43.546 ± 0.200 | 33.571 ± 0.447 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `128x128` | - | 0.259 ± 0.005 | 0.128 ± 0.001 | 0.200 ± 0.007 | 0.221 ± 0.003 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `256x256` | - | 0.812 ± 0.007 | 0.534 ± 0.001 | 0.823 ± 0.008 | 0.814 ± 0.018 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `512x512` | - | 4.731 ± 0.030 | 3.561 ± 0.033 | 7.023 ± 0.112 | 5.021 ± 0.173 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 8 | `64x64` | - | 0.128 ± 0.003 | 0.053 ± 0.000 | 0.089 ± 0.003 | 0.083 ± 0.001 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `1024x1024` | - | 197.271 ± 4.837 | 95.086 ± 1.032 | 111.006 ± 1.639 | 181.262 ± 16.016 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `256x256` | - | 8.615 ± 0.151 | 3.522 ± 0.018 | 3.429 ± 0.010 | 9.633 ± 1.509 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 8 | `512x512` | - | 31.977 ± 0.105 | 15.839 ± 0.094 | 19.869 ± 0.150 | 40.602 ± 11.086 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `1024x1024` | - | 159.971 ± 0.906 | 95.178 ± 0.496 | 104.357 ± 0.858 | 183.955 ± 9.599 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `256x256` | - | 5.246 ± 0.036 | 3.490 ± 0.017 | 3.647 ± 0.026 | 10.402 ± 2.731 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 8 | `512x512` | - | 25.416 ± 0.071 | 15.676 ± 0.017 | 21.034 ± 0.424 | 42.565 ± 2.283 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `1024x1024,rhs=1` | - | 8.517 ± 0.125 | 20.865 ± 0.062 | 129.451 ± 78.884 | 134.223 ± 31.949 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `128x128,rhs=1` | - | 0.247 ± 0.005 | 0.385 ± 0.001 | 0.275 ± 0.006 | 0.225 ± 0.002 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `256x256,rhs=1` | - | 0.563 ± 0.014 | 1.172 ± 0.009 | 28.901 ± 2.021 | 2.401 ± 21.305 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `512x512,rhs=1` | - | 1.959 ± 0.027 | 4.387 ± 0.048 | 59.703 ± 53.999 | 134.739 ± 144.026 | - | - |
| large | `grad_sum_solve_backward` | f64 | 8 | `64x64,rhs=1` | - | 0.143 ± 0.003 | 0.140 ± 0.001 | 0.116 ± 0.003 | 0.070 ± 0.001 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `1024x1024,rhs=1` | - | 21.917 ± 0.549 | 13.913 ± 0.134 | 9.341 ± 0.186 | 14.836 ± 1.520 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `256x256,rhs=1` | - | 1.577 ± 0.104 | 0.773 ± 0.005 | 0.783 ± 0.042 | 0.578 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 8 | `512x512,rhs=1` | - | 5.172 ± 0.173 | 2.976 ± 0.019 | 2.365 ± 0.038 | 3.671 ± 0.051 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `1024x1024,rhs=1` | - | 15.095 ± 0.543 | 13.500 ± 0.097 | 15.233 ± 0.135 | 14.293 ± 1.864 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `256x256,rhs=1` | - | 0.986 ± 0.016 | 0.776 ± 0.003 | 0.891 ± 0.032 | 0.589 ± 0.014 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 8 | `512x512,rhs=1` | - | 3.465 ± 0.082 | 2.905 ± 0.015 | 3.306 ± 0.021 | 3.635 ± 0.095 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `1024x1024` | - | 218.335 ± 3.826 | 324.859 ± 0.875 | 301.741 ± 2.479 | 460.985 ± 8.743 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `128x128` | - | 3.501 ± 0.021 | 4.937 ± 0.012 | 3.086 ± 0.028 | 4.213 ± 0.053 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `256x256` | - | 19.295 ± 0.133 | 27.469 ± 0.189 | 14.206 ± 0.032 | 19.465 ± 2.321 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `512x512` | - | 42.064 ± 0.153 | 68.324 ± 0.214 | 72.329 ± 2.740 | 103.855 ± 4.829 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 8 | `64x64` | - | 0.961 ± 0.006 | 1.227 ± 0.004 | 0.799 ± 0.007 | 0.828 ± 0.051 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `1024x1024` | - | 625.106 ± 1.799 | 301.961 ± 1.319 | 326.793 ± 5.452 | 460.735 ± 18.142 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `256x256` | - | 56.651 ± 0.057 | 27.517 ± 0.244 | 15.135 ± 0.311 | 19.984 ± 1.628 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 8 | `512x512` | - | 119.545 ± 0.233 | 68.119 ± 0.229 | 68.777 ± 0.407 | 106.513 ± 4.361 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `1024x1024` | - | 407.893 ± 1.425 | 324.704 ± 0.448 | 275.797 ± 6.173 | 462.670 ± 13.096 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `256x256` | - | 37.699 ± 0.327 | 27.376 ± 0.026 | 14.053 ± 0.041 | 19.791 ± 1.489 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 8 | `512x512` | - | 80.657 ± 0.096 | 68.343 ± 0.366 | 63.918 ± 0.308 | 102.049 ± 4.467 | - | - |
| large | `matmul` | f64 | 8 | `1024x1024` | - | 8.997 ± 0.030 | 8.978 ± 0.016 | 12.721 ± 0.074 | 11.863 ± 1.073 | - | - |
| large | `matmul` | f64 | 8 | `128x128` | - | 0.040 ± 0.000 | 0.040 ± 0.005 | 0.070 ± 0.001 | 0.079 ± 0.010 | - | - |
| large | `matmul` | f64 | 8 | `256x256` | - | 0.176 ± 0.001 | 0.173 ± 0.001 | 0.294 ± 0.004 | 0.319 ± 0.017 | - | - |
| large | `matmul` | f64 | 8 | `512x512` | - | 1.154 ± 0.004 | 1.163 ± 0.005 | 2.246 ± 0.027 | 1.777 ± 0.041 | - | - |
| large | `matmul_rect` | f64 | 8 | `1024x256 * 256x1024` | - | 2.278 ± 0.007 | 2.274 ± 0.004 | 3.497 ± 0.032 | 3.184 ± 0.276 | - | - |
| large | `matmul_rect` | f64 | 8 | `256x1024 * 1024x256` | - | 0.671 ± 0.008 | 0.677 ± 0.010 | 0.922 ± 0.010 | 1.109 ± 0.037 | - | - |
| large | `qr` | f64 | 8 | `1024x1024` | - | 26.142 ± 0.083 | 25.998 ± 0.086 | 49.729 ± 0.264 | 70.320 ± 3.133 | - | - |
| large | `qr` | f64 | 8 | `128x128` | - | 0.559 ± 0.001 | 0.558 ± 0.004 | 0.448 ± 0.012 | 0.888 ± 0.011 | - | - |
| large | `qr` | f64 | 8 | `256x256` | - | 1.217 ± 0.009 | 1.214 ± 0.010 | 1.873 ± 0.027 | 3.323 ± 0.117 | - | - |
| large | `qr` | f64 | 8 | `512x512` | - | 4.840 ± 0.016 | 4.833 ± 0.014 | 9.470 ± 0.038 | 16.646 ± 0.302 | - | - |
| large | `qr` | f64 | 8 | `64x64` | - | 0.136 ± 0.000 | 0.135 ± 0.000 | 0.160 ± 0.001 | 0.121 ± 0.003 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=1` | - | 6.453 ± 0.015 | 6.176 ± 0.073 | 131.791 ± 45.506 | 133.162 ± 15.693 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=16` | - | 7.020 ± 0.022 | 6.655 ± 0.133 | 116.520 ± 74.259 | 128.212 ± 4.371 | - | - |
| large | `solve` | f64 | 8 | `1024x1024,rhs=64` | - | 8.841 ± 0.041 | 8.265 ± 0.046 | 116.970 ± 16.803 | 113.534 ± 15.723 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=1` | - | 0.112 ± 0.001 | 0.108 ± 0.000 | 0.180 ± 0.002 | 0.188 ± 0.005 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=16` | - | 0.139 ± 0.001 | 0.125 ± 0.002 | 8.875 ± 1.707 | 2.117 ± 0.116 | - | - |
| large | `solve` | f64 | 8 | `128x128,rhs=64` | - | 0.226 ± 0.002 | 0.178 ± 0.002 | 8.291 ± 1.308 | 0.306 ± 0.010 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=1` | - | 0.377 ± 0.002 | 0.352 ± 0.004 | 22.011 ± 4.011 | 19.752 ± 16.944 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=16` | - | 0.453 ± 0.007 | 0.415 ± 0.005 | 34.694 ± 2.971 | 1.362 ± 0.568 | - | - |
| large | `solve` | f64 | 8 | `256x256,rhs=64` | - | 0.677 ± 0.004 | 0.575 ± 0.009 | 32.822 ± 4.040 | 12.805 ± 71.255 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=1` | - | 1.452 ± 0.007 | 1.366 ± 0.047 | 32.837 ± 59.992 | 68.056 ± 142.737 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=16` | - | 1.704 ± 0.010 | 1.497 ± 0.007 | 45.155 ± 72.954 | 131.840 ± 134.044 | - | - |
| large | `solve` | f64 | 8 | `512x512,rhs=64` | - | 2.428 ± 0.007 | 2.054 ± 0.020 | 34.134 ± 31.895 | 61.088 ± 119.762 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=1` | - | 0.036 ± 0.000 | 0.037 ± 0.000 | 0.053 ± 0.002 | 0.053 ± 0.001 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=16` | - | 0.046 ± 0.001 | 0.041 ± 0.001 | 0.066 ± 0.001 | 0.096 ± 0.002 | - | - |
| large | `solve` | f64 | 8 | `64x64,rhs=64` | - | 0.084 ± 0.000 | 0.059 ± 0.001 | 0.076 ± 0.002 | 0.123 ± 0.004 | - | - |
| large | `svd` | f64 | 8 | `1024x1024` | - | 192.308 ± 0.725 | 177.182 ± 1.312 | 263.687 ± 1.109 | 378.249 ± 7.674 | - | - |
| large | `svd` | f64 | 8 | `128x128` | - | 3.319 ± 0.010 | 3.320 ± 0.012 | 2.824 ± 0.010 | 3.341 ± 0.015 | - | - |
| large | `svd` | f64 | 8 | `256x256` | - | 18.116 ± 0.029 | 18.119 ± 0.106 | 13.293 ± 0.041 | 15.600 ± 0.027 | - | - |
| large | `svd` | f64 | 8 | `512x512` | - | 38.593 ± 0.063 | 38.722 ± 0.084 | 54.212 ± 0.115 | 87.676 ± 0.947 | - | - |
| large | `svd` | f64 | 8 | `64x64` | - | 0.838 ± 0.001 | 0.827 ± 0.003 | 0.706 ± 0.004 | 0.756 ± 0.006 | - | - |
| small | `eigh` | f64 | 8 | `16x16` | - | 0.037 ± 0.000 | 0.036 ± 0.000 | 0.034 ± 0.001 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 8 | `32x32` | - | 0.122 ± 0.001 | 0.123 ± 0.001 | 0.113 ± 0.000 | 0.126 ± 0.002 | - | - |
| small | `eigh` | f64 | 8 | `4x4` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `eigh` | f64 | 8 | `8x8` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `16x16` | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.153 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.145 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `32x32` | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.022 ± 0.001 | 0.158 ± 0.004 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.000 | 0.148 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.154 ± 0.003 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `16x16` | - | 0.361 ± 0.004 | 0.089 ± 0.000 | 0.160 ± 0.005 | 0.042 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `2x2` | - | 0.255 ± 0.002 | 0.035 ± 0.000 | 0.124 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `32x32` | - | 0.646 ± 0.006 | 0.244 ± 0.001 | 0.259 ± 0.006 | 0.131 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `4x4` | - | 0.266 ± 0.001 | 0.039 ± 0.000 | 0.126 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 8 | `8x8` | - | 0.288 ± 0.004 | 0.051 ± 0.000 | 0.133 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `16x16` | - | 0.146 ± 0.002 | 0.088 ± 0.000 | 0.135 ± 0.001 | 0.043 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `2x2` | - | 0.082 ± 0.001 | 0.034 ± 0.000 | 0.104 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `32x32` | - | 0.329 ± 0.002 | 0.242 ± 0.000 | 0.224 ± 0.006 | 0.133 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `4x4` | - | 0.087 ± 0.001 | 0.038 ± 0.000 | 0.107 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 8 | `8x8` | - | 0.100 ± 0.001 | 0.049 ± 0.000 | 0.112 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `16x16` | - | 0.982 ± 0.007 | 0.055 ± 0.000 | 0.230 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `2x2` | - | 0.928 ± 0.005 | 0.045 ± 0.000 | 0.213 ± 0.010 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `32x32` | - | 1.030 ± 0.019 | 0.082 ± 0.000 | 0.338 ± 0.008 | 0.248 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `4x4` | - | 0.953 ± 0.008 | 0.046 ± 0.000 | 0.209 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 8 | `8x8` | - | 0.970 ± 0.012 | 0.048 ± 0.000 | 0.211 ± 0.004 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `16x16` | - | 0.218 ± 0.002 | 0.052 ± 0.000 | 0.180 ± 0.005 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `2x2` | - | 0.201 ± 0.003 | 0.041 ± 0.000 | 0.164 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `32x32` | - | 0.269 ± 0.004 | 0.080 ± 0.000 | 0.223 ± 0.005 | 0.060 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `4x4` | - | 0.200 ± 0.001 | 0.042 ± 0.000 | 0.164 ± 0.005 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 8 | `8x8` | - | 0.205 ± 0.001 | 0.044 ± 0.000 | 0.166 ± 0.005 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `16x16` | - | 0.066 ± 0.002 | 0.017 ± 0.000 | 0.047 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `2x2` | - | 0.062 ± 0.001 | 0.016 ± 0.000 | 0.041 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `32x32` | - | 0.076 ± 0.002 | 0.024 ± 0.000 | 0.065 ± 0.003 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `4x4` | - | 0.062 ± 0.002 | 0.016 ± 0.000 | 0.043 ± 0.001 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 8 | `8x8` | - | 0.063 ± 0.001 | 0.016 ± 0.000 | 0.044 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `16x16` | - | 1.372 ± 0.020 | 0.081 ± 0.000 | 0.164 ± 0.006 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `2x2` | - | 1.268 ± 0.008 | 0.057 ± 0.000 | 0.145 ± 0.002 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `32x32` | - | 1.457 ± 0.034 | 0.145 ± 0.001 | 0.216 ± 0.008 | 0.244 ± 0.226 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `4x4` | - | 1.315 ± 0.015 | 0.058 ± 0.000 | 0.144 ± 0.003 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 8 | `8x8` | - | 1.323 ± 0.006 | 0.063 ± 0.000 | 0.149 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `16x16` | - | 0.209 ± 0.002 | 0.078 ± 0.000 | 0.166 ± 0.006 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `2x2` | - | 0.180 ± 0.002 | 0.055 ± 0.000 | 0.149 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `32x32` | - | 0.296 ± 0.002 | 0.142 ± 0.001 | 0.234 ± 0.005 | 0.091 ± 0.020 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `4x4` | - | 0.182 ± 0.003 | 0.057 ± 0.000 | 0.150 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 8 | `8x8` | - | 0.188 ± 0.002 | 0.061 ± 0.000 | 0.154 ± 0.001 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `16x16,rhs=1` | - | 0.089 ± 0.001 | 0.059 ± 0.000 | 0.070 ± 0.003 | 0.023 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `2x2,rhs=1` | - | 0.085 ± 0.001 | 0.050 ± 0.000 | 0.065 ± 0.000 | 0.017 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `32x32,rhs=1` | - | 0.101 ± 0.002 | 0.074 ± 0.001 | 0.082 ± 0.001 | 0.024 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `4x4,rhs=1` | - | 0.086 ± 0.002 | 0.050 ± 0.000 | 0.066 ± 0.001 | 0.019 ± 0.002 | - | - |
| small | `grad_sum_solve_backward` | f64 | 8 | `8x8,rhs=1` | - | 0.088 ± 0.001 | 0.053 ± 0.000 | 0.067 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `16x16,rhs=1` | - | 0.202 ± 0.001 | 0.040 ± 0.000 | 0.336 ± 0.003 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `2x2,rhs=1` | - | 0.187 ± 0.005 | 0.035 ± 0.000 | 0.324 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `32x32,rhs=1` | - | 0.222 ± 0.002 | 0.050 ± 0.000 | 0.351 ± 0.008 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `4x4,rhs=1` | - | 0.190 ± 0.003 | 0.035 ± 0.000 | 0.327 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 8 | `8x8,rhs=1` | - | 0.194 ± 0.002 | 0.036 ± 0.000 | 0.328 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `16x16,rhs=1` | - | 0.101 ± 0.001 | 0.039 ± 0.000 | 0.146 ± 0.002 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `2x2,rhs=1` | - | 0.094 ± 0.001 | 0.034 ± 0.000 | 0.139 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `32x32,rhs=1` | - | 0.116 ± 0.001 | 0.050 ± 0.000 | 0.168 ± 0.007 | 0.020 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `4x4,rhs=1` | - | 0.097 ± 0.004 | 0.034 ± 0.000 | 0.139 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 8 | `8x8,rhs=1` | - | 0.096 ± 0.001 | 0.036 ± 0.000 | 0.142 ± 0.002 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `16x16` | - | 0.130 ± 0.007 | 0.100 ± 0.000 | 0.108 ± 0.001 | 0.072 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `2x2` | - | 0.070 ± 0.002 | 0.024 ± 0.000 | 0.047 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `32x32` | - | 0.293 ± 0.005 | 0.315 ± 0.002 | 0.269 ± 0.002 | 0.206 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `4x4` | - | 0.074 ± 0.001 | 0.029 ± 0.000 | 0.053 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 8 | `8x8` | - | 0.085 ± 0.002 | 0.045 ± 0.000 | 0.067 ± 0.000 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `16x16` | - | 0.399 ± 0.002 | 0.103 ± 0.000 | 0.260 ± 0.005 | 0.069 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `2x2` | - | 0.219 ± 0.003 | 0.025 ± 0.000 | 0.185 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `32x32` | - | 0.876 ± 0.003 | 0.314 ± 0.001 | 0.450 ± 0.007 | 0.201 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `4x4` | - | 0.232 ± 0.003 | 0.031 ± 0.000 | 0.191 ± 0.005 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 8 | `8x8` | - | 0.267 ± 0.002 | 0.046 ± 0.000 | 0.211 ± 0.004 | 0.022 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `16x16` | - | 0.193 ± 0.002 | 0.101 ± 0.000 | 0.176 ± 0.005 | 0.070 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `2x2` | - | 0.082 ± 0.002 | 0.024 ± 0.000 | 0.114 ± 0.001 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `32x32` | - | 0.503 ± 0.003 | 0.314 ± 0.001 | 0.356 ± 0.014 | 0.204 ± 0.003 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `4x4` | - | 0.089 ± 0.002 | 0.029 ± 0.000 | 0.118 ± 0.001 | 0.019 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 8 | `8x8` | - | 0.110 ± 0.002 | 0.045 ± 0.000 | 0.133 ± 0.001 | 0.022 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `16x16` | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `2x2` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `32x32` | - | 0.008 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| small | `matmul` | f64 | 8 | `4x4` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 8 | `64x64` | - | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.024 ± 0.001 | 0.037 ± 0.003 | - | - |
| small | `matmul` | f64 | 8 | `8x8` | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `16x16` | - | 0.016 ± 0.000 | 0.015 ± 0.000 | 0.015 ± 0.001 | 0.020 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `2x2` | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `32x32` | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.036 ± 0.001 | 0.038 ± 0.001 | - | - |
| small | `qr` | f64 | 8 | `4x4` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 8 | `8x8` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=1` | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `16x16,rhs=4` | - | 0.011 ± 0.001 | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=1` | - | 0.016 ± 0.001 | 0.017 ± 0.000 | 0.029 ± 0.001 | 0.015 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `32x32,rhs=4` | - | 0.016 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.001 | 0.016 ± 0.002 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=1` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 8 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=1` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `solve` | f64 | 8 | `8x8,rhs=4` | - | 0.009 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `svd` | f64 | 8 | `16x16` | - | 0.062 ± 0.000 | 0.061 ± 0.000 | 0.065 ± 0.000 | 0.073 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `2x2` | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| small | `svd` | f64 | 8 | `32x32` | - | 0.213 ± 0.001 | 0.213 ± 0.001 | 0.222 ± 0.004 | 0.204 ± 0.001 | - | - |
| small | `svd` | f64 | 8 | `4x4` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.002 | - | - |
| small | `svd` | f64 | 8 | `8x8` | - | 0.022 ± 0.000 | 0.021 ± 0.000 | 0.026 ± 0.000 | 0.029 ± 0.001 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-eager` is 26.6x faster than the slowest successful cell (`jax-cpu`, 1.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`jax-cpu`, 1.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-eager` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`jax-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-eager` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.151 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-eager` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch1024 (native batch layout)`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`jax-cpu`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 19.1x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 16.3x faster than the slowest successful cell (`jax-cpu`, 0.470 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 16.5x faster than the slowest successful cell (`jax-cpu`, 0.470 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 18.2x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 19.4x faster than the slowest successful cell (`jax-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 13.3x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch1024 (native batch layout)`): `tenferro-trace` is 41.3x faster than the slowest successful cell (`pytorch-cpu`, 9.412 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.259 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch256 (native batch layout)`): `tenferro-trace` is 33.4x faster than the slowest successful cell (`pytorch-cpu`, 2.580 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`16x16xbatch64 (native batch layout)`): `tenferro-trace` is 19.6x faster than the slowest successful cell (`pytorch-cpu`, 0.722 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 20.2x faster than the slowest successful cell (`tenferro-eager`, 0.535 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-eager` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 7.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 67.2x faster than the slowest successful cell (`pytorch-cpu`, 7.132 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-eager` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 1.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 42.2x faster than the slowest successful cell (`pytorch-cpu`, 1.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=8, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 18.4x faster than the slowest successful cell (`pytorch-cpu`, 0.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`256x256`): `tenferro-eager` is 16.0x faster than the slowest successful cell (`pytorch-cpu`, 150.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/eigh` (f64, threads=8, shape=`256x256`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 150.789 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-eager` is 15.8x faster than the slowest successful cell (`jax-cpu`, 134.223 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 28.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-eager` is 51.3x faster than the slowest successful cell (`pytorch-cpu`, 28.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 24.7x faster than the slowest successful cell (`pytorch-cpu`, 28.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-eager` is 68.8x faster than the slowest successful cell (`jax-cpu`, 134.739 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/grad_sum_solve_backward` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 30.7x faster than the slowest successful cell (`jax-cpu`, 134.739 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-eager` is 20.6x faster than the slowest successful cell (`jax-cpu`, 133.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=1`): `tenferro-trace` is 21.6x faster than the slowest successful cell (`jax-cpu`, 133.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-eager` is 18.3x faster than the slowest successful cell (`jax-cpu`, 128.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=16`): `tenferro-trace` is 19.3x faster than the slowest successful cell (`jax-cpu`, 128.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-eager` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 116.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`1024x1024,rhs=64`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 116.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-eager` is 63.8x faster than the slowest successful cell (`pytorch-cpu`, 8.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=16`): `tenferro-trace` is 70.8x faster than the slowest successful cell (`pytorch-cpu`, 8.875 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `jax-cpu` is 27.1x faster than the slowest successful cell (`pytorch-cpu`, 8.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-eager` is 36.7x faster than the slowest successful cell (`pytorch-cpu`, 8.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`128x128,rhs=64`): `tenferro-trace` is 46.7x faster than the slowest successful cell (`pytorch-cpu`, 8.291 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-eager` is 58.4x faster than the slowest successful cell (`pytorch-cpu`, 22.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=1`): `tenferro-trace` is 62.6x faster than the slowest successful cell (`pytorch-cpu`, 22.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `jax-cpu` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 34.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-eager` is 76.6x faster than the slowest successful cell (`pytorch-cpu`, 34.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=16`): `tenferro-trace` is 83.6x faster than the slowest successful cell (`pytorch-cpu`, 34.694 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-eager` is 48.5x faster than the slowest successful cell (`pytorch-cpu`, 32.822 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`256x256,rhs=64`): `tenferro-trace` is 57.1x faster than the slowest successful cell (`pytorch-cpu`, 32.822 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-eager` is 46.9x faster than the slowest successful cell (`jax-cpu`, 68.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=1`): `tenferro-trace` is 49.8x faster than the slowest successful cell (`jax-cpu`, 68.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-eager` is 77.4x faster than the slowest successful cell (`jax-cpu`, 131.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=16`): `tenferro-trace` is 88.1x faster than the slowest successful cell (`jax-cpu`, 131.840 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-eager` is 25.2x faster than the slowest successful cell (`jax-cpu`, 61.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `large/solve` (f64, threads=8, shape=`512x512,rhs=64`): `tenferro-trace` is 29.7x faster than the slowest successful cell (`jax-cpu`, 61.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-eager` is 25.9x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `pytorch-cpu` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-eager` is 30.1x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-eager` is 18.0x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.158 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-eager` is 30.2x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`jax-cpu`, 0.148 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-eager` is 30.5x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 20.9x faster than the slowest successful cell (`tenferro-eager`, 0.255 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 17.5x faster than the slowest successful cell (`tenferro-eager`, 0.266 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 18.2x faster than the slowest successful cell (`tenferro-eager`, 0.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 48.1x faster than the slowest successful cell (`tenferro-eager`, 0.982 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`tenferro-eager`, 0.982 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 66.2x faster than the slowest successful cell (`tenferro-eager`, 0.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 20.5x faster than the slowest successful cell (`tenferro-eager`, 0.928 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 12.6x faster than the slowest successful cell (`tenferro-eager`, 1.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 62.4x faster than the slowest successful cell (`tenferro-eager`, 0.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 20.7x faster than the slowest successful cell (`tenferro-eager`, 0.953 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 44.6x faster than the slowest successful cell (`tenferro-eager`, 0.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`tenferro-eager`, 0.970 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 0.218 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 0.201 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.200 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 10.0x faster than the slowest successful cell (`tenferro-eager`, 0.205 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 71.5x faster than the slowest successful cell (`tenferro-eager`, 1.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`16x16`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`tenferro-eager`, 1.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 94.4x faster than the slowest successful cell (`tenferro-eager`, 1.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`2x2`): `tenferro-trace` is 22.2x faster than the slowest successful cell (`tenferro-eager`, 1.268 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`32x32`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 1.457 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 84.6x faster than the slowest successful cell (`tenferro-eager`, 1.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`4x4`): `tenferro-trace` is 22.5x faster than the slowest successful cell (`tenferro-eager`, 1.315 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 65.2x faster than the slowest successful cell (`tenferro-eager`, 1.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=8, shape=`8x8`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`tenferro-eager`, 1.323 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`16x16`): `jax-cpu` is 11.6x faster than the slowest successful cell (`tenferro-eager`, 0.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-eager`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`16x16,rhs=1`): `jax-cpu` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.336 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`2x2,rhs=1`): `jax-cpu` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.324 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`32x32,rhs=1`): `jax-cpu` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.351 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.327 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=8, shape=`8x8,rhs=1`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`2x2`): `jax-cpu` is 15.9x faster than the slowest successful cell (`tenferro-eager`, 0.219 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`4x4`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-eager`, 0.232 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=8, shape=`8x8`): `jax-cpu` is 12.4x faster than the slowest successful cell (`tenferro-eager`, 0.267 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

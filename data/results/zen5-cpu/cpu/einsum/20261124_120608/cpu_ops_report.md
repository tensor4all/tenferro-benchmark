# CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

## Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

## Threads: 1

- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_120608/cpu_ops_t1_20261124_120608.csv`

## CPU Benchmark Items

Small rows use time/memory-bounded batches, normalized per operation. Tenferro eager/trace share a CPU execution scope outside timing. Eager primal and backward rows reuse a borrowed session entered before sampling. A backward row times the forward pass plus `EagerTensor::backward` run through that session's engine, one operation per sample; its gradients are checked against scalar references before timing, and accumulated gradients are cleared after every sample, outside timing. Raw JSONL records contain batch durations and operation counts.

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 33.266 ± 0.154 | 33.099 ± 0.144 | 27.296 ± 0.347 | 34.048 ± 0.246 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.511 ± 0.001 | 0.506 ± 0.001 | 0.409 ± 0.002 | 0.550 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 8.305 ± 0.021 | 8.222 ± 0.007 | 6.858 ± 0.060 | 8.567 ± 0.074 | - | - |
| batched | `batched_eigh` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 2.078 ± 0.019 | 2.048 ± 0.003 | 1.684 ± 0.009 | 2.149 ± 0.020 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.546 ± 0.007 | 0.540 ± 0.012 | 0.538 ± 0.005 | 0.431 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.017 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.142 ± 0.001 | 0.139 ± 0.001 | 0.141 ± 0.001 | 0.118 ± 0.002 | - | - |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.040 ± 0.000 | 0.038 ± 0.000 | 0.042 ± 0.001 | 0.046 ± 0.001 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 2.853 ± 0.004 | 2.853 ± 0.006 | 2.711 ± 0.006 | 2.598 ± 0.025 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.050 ± 0.000 | 0.048 ± 0.000 | 0.051 ± 0.000 | 0.051 ± 0.000 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.719 ± 0.001 | 0.718 ± 0.004 | 0.685 ± 0.004 | 0.664 ± 0.006 | - | - |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.184 ± 0.001 | 0.183 ± 0.001 | 0.177 ± 0.002 | 0.175 ± 0.005 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 8.805 ± 0.019 | 8.794 ± 0.029 | 7.995 ± 0.097 | 8.884 ± 0.060 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.141 ± 0.000 | 0.139 ± 0.000 | 0.132 ± 0.002 | 0.154 ± 0.003 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 2.212 ± 0.011 | 2.205 ± 0.013 | 2.003 ± 0.008 | 2.230 ± 0.007 | - | - |
| batched | `batched_eigh` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.553 ± 0.003 | 0.551 ± 0.001 | 0.505 ± 0.003 | 0.569 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 0.440 ± 0.005 | 0.440 ± 0.005 | 1.151 ± 0.022 | 1.095 ± 0.040 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.012 ± 0.000 | 0.011 ± 0.000 | 0.035 ± 0.001 | 0.182 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.115 ± 0.001 | 0.114 ± 0.002 | 0.295 ± 0.005 | 0.398 ± 0.017 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.032 ± 0.000 | 0.031 ± 0.000 | 0.084 ± 0.004 | 0.222 ± 0.005 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.148 ± 0.001 | 0.147 ± 0.000 | 0.046 ± 0.001 | 0.182 ± 0.010 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.000 | 0.149 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.040 ± 0.000 | 0.040 ± 0.000 | 0.023 ± 0.001 | 0.172 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.013 ± 0.000 | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.166 ± 0.001 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.160 ± 0.002 | 0.156 ± 0.001 | 0.166 ± 0.002 | 0.206 ± 0.007 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.000 | 0.170 ± 0.004 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.044 ± 0.001 | 0.042 ± 0.000 | 0.054 ± 0.001 | 0.176 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.015 ± 0.001 | 0.014 ± 0.000 | 0.025 ± 0.001 | 0.171 ± 0.002 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 0.180 ± 0.002 | 0.178 ± 0.007 | 0.256 ± 0.004 | 0.323 ± 0.006 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.019 ± 0.001 | 0.168 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.048 ± 0.000 | 0.047 ± 0.000 | 0.075 ± 0.001 | 0.205 ± 0.003 | - | - |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.016 ± 0.000 | 0.016 ± 0.000 | 0.032 ± 0.001 | 0.178 ± 0.002 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 9.579 ± 0.024 | 9.772 ± 0.056 | 5.897 ± 0.049 | 8.498 ± 0.120 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.160 ± 0.002 | 0.156 ± 0.001 | 0.100 ± 0.002 | 0.144 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 2.445 ± 0.008 | 2.448 ± 0.017 | 1.496 ± 0.018 | 2.122 ± 0.177 | - | - |
| batched | `batched_qr` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.618 ± 0.006 | 0.603 ± 0.003 | 0.376 ± 0.006 | 0.538 ± 0.008 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.319 ± 0.002 | 0.317 ± 0.001 | 0.269 ± 0.003 | 0.341 ± 0.005 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.085 ± 0.000 | 0.083 ± 0.000 | 0.073 ± 0.001 | 0.095 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.026 ± 0.000 | 0.024 ± 0.000 | 0.023 ± 0.000 | 0.031 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.807 ± 0.002 | 0.814 ± 0.001 | 0.691 ± 0.005 | 0.958 ± 0.010 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.019 ± 0.000 | 0.018 ± 0.000 | 0.017 ± 0.001 | 0.026 ± 0.000 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.207 ± 0.001 | 0.204 ± 0.000 | 0.179 ± 0.003 | 0.249 ± 0.004 | - | - |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.057 ± 0.000 | 0.055 ± 0.000 | 0.049 ± 0.001 | 0.078 ± 0.003 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 2.571 ± 0.019 | 2.553 ± 0.005 | 1.741 ± 0.007 | 2.736 ± 0.070 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.046 ± 0.000 | 0.044 ± 0.000 | 0.034 ± 0.001 | 0.061 ± 0.001 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.643 ± 0.001 | 0.642 ± 0.002 | 0.445 ± 0.004 | 0.697 ± 0.006 | - | - |
| batched | `batched_qr` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.166 ± 0.000 | 0.164 ± 0.002 | 0.115 ± 0.001 | 0.183 ± 0.003 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 2.221 ± 0.023 | 2.244 ± 0.011 | 1.880 ± 0.019 | 2.962 ± 0.071 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.043 ± 0.000 | 0.043 ± 0.000 | 0.051 ± 0.002 | 0.069 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.577 ± 0.002 | 0.558 ± 0.003 | 0.508 ± 0.016 | 0.748 ± 0.018 | - | - |
| batched | `batched_solve` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.151 ± 0.002 | 0.142 ± 0.000 | 0.140 ± 0.006 | 0.203 ± 0.008 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.213 ± 0.002 | 0.166 ± 0.005 | 0.108 ± 0.002 | 0.344 ± 0.004 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.000 | 0.016 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.060 ± 0.000 | 0.048 ± 0.000 | 0.043 ± 0.001 | 0.098 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.019 ± 0.000 | 0.025 ± 0.001 | 0.041 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.312 ± 0.001 | 0.270 ± 0.001 | 0.158 ± 0.001 | 0.494 ± 0.005 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.013 ± 0.000 | 0.013 ± 0.000 | 0.021 ± 0.000 | 0.020 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.084 ± 0.000 | 0.075 ± 0.000 | 0.054 ± 0.001 | 0.135 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.027 ± 0.000 | 0.025 ± 0.000 | 0.029 ± 0.001 | 0.051 ± 0.001 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 0.823 ± 0.003 | 0.787 ± 0.002 | 0.561 ± 0.006 | 1.032 ± 0.012 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.021 ± 0.000 | 0.021 ± 0.000 | 0.028 ± 0.001 | 0.036 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.214 ± 0.002 | 0.199 ± 0.001 | 0.154 ± 0.003 | 0.267 ± 0.002 | - | - |
| batched | `batched_solve` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.060 ± 0.000 | 0.057 ± 0.001 | 0.054 ± 0.001 | 0.083 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 57.885 ± 0.091 | 58.078 ± 0.185 | 60.075 ± 0.367 | 59.489 ± 0.283 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.907 ± 0.006 | 0.896 ± 0.005 | 0.923 ± 0.010 | 0.955 ± 0.003 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 14.425 ± 0.076 | 14.429 ± 0.059 | 15.099 ± 0.164 | 15.090 ± 0.925 | - | - |
| batched | `batched_svd` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 3.620 ± 0.038 | 3.600 ± 0.018 | 3.748 ± 0.014 | 3.780 ± 0.038 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 1.302 ± 0.008 | 1.303 ± 0.004 | 1.484 ± 0.014 | 1.556 ± 0.027 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.027 ± 0.000 | 0.026 ± 0.000 | 0.033 ± 0.001 | 0.037 ± 0.000 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.334 ± 0.002 | 0.330 ± 0.001 | 0.377 ± 0.002 | 0.407 ± 0.010 | - | - |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.088 ± 0.000 | 0.087 ± 0.000 | 0.102 ± 0.003 | 0.109 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 4.682 ± 0.012 | 4.690 ± 0.009 | 5.606 ± 0.024 | 5.029 ± 0.049 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.079 ± 0.000 | 0.078 ± 0.000 | 0.097 ± 0.001 | 0.091 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 1.178 ± 0.006 | 1.181 ± 0.014 | 1.418 ± 0.015 | 1.276 ± 0.008 | - | - |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.300 ± 0.002 | 0.299 ± 0.001 | 0.358 ± 0.001 | 0.338 ± 0.005 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 15.226 ± 0.145 | 15.055 ± 0.042 | 18.794 ± 0.093 | 15.954 ± 0.127 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.240 ± 0.001 | 0.238 ± 0.000 | 0.299 ± 0.001 | 0.265 ± 0.001 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 3.766 ± 0.008 | 3.784 ± 0.027 | 4.709 ± 0.020 | 4.023 ± 0.017 | - | - |
| batched | `batched_svd` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.947 ± 0.001 | 0.944 ± 0.003 | 1.182 ± 0.005 | 1.019 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout)` | - | 2.894 ± 0.117 | 1.674 ± 0.007 | 10.326 ± 0.097 | 2.998 ± 0.067 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch16 (native batch layout)` | - | 0.101 ± 0.002 | 0.040 ± 0.000 | 0.270 ± 0.006 | 0.066 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch256 (native batch layout)` | - | 0.706 ± 0.003 | 0.428 ± 0.001 | 2.732 ± 0.104 | 0.736 ± 0.009 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `16x16xbatch64 (native batch layout)` | - | 0.225 ± 0.003 | 0.116 ± 0.001 | 0.757 ± 0.028 | 0.188 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout)` | - | 0.529 ± 0.005 | 0.484 ± 0.003 | 0.195 ± 0.006 | 0.045 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | - | 0.066 ± 0.002 | 0.021 ± 0.000 | 0.105 ± 0.001 | 0.013 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch256 (native batch layout)` | - | 0.177 ± 0.001 | 0.131 ± 0.001 | 0.126 ± 0.002 | 0.026 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | - | 0.089 ± 0.002 | 0.044 ± 0.000 | 0.109 ± 0.001 | 0.026 ± 0.001 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout)` | - | 0.613 ± 0.002 | 0.513 ± 0.003 | 0.577 ± 0.015 | 0.562 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | - | 0.071 ± 0.003 | 0.022 ± 0.000 | 0.110 ± 0.001 | 0.025 ± 0.003 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch256 (native batch layout)` | - | 0.197 ± 0.002 | 0.139 ± 0.000 | 0.223 ± 0.006 | 0.155 ± 0.007 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | - | 0.096 ± 0.001 | 0.046 ± 0.000 | 0.131 ± 0.003 | 0.056 ± 0.002 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout)` | - | 1.187 ± 0.024 | 0.648 ± 0.002 | 7.375 ± 0.124 | 0.980 ± 0.004 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch16 (native batch layout)` | - | 0.074 ± 0.002 | 0.024 ± 0.000 | 0.220 ± 0.002 | 0.040 ± 0.008 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch256 (native batch layout)` | - | 0.293 ± 0.027 | 0.172 ± 0.001 | 1.911 ± 0.036 | 0.255 ± 0.005 | - | - |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `8x8xbatch64 (native batch layout)` | - | 0.115 ± 0.002 | 0.054 ± 0.000 | 0.563 ± 0.007 | 0.085 ± 0.003 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch1024 (native batch layout),rhs=1` | - | 3.509 ± 0.034 | 8.398 ± 0.031 | 4.257 ± 0.157 | 4.926 ± 0.116 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch16 (native batch layout),rhs=1` | - | 0.141 ± 0.003 | 0.170 ± 0.001 | 0.140 ± 0.003 | 0.109 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch256 (native batch layout),rhs=1` | - | 0.950 ± 0.016 | 2.118 ± 0.008 | 1.062 ± 0.040 | 1.240 ± 0.016 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `16x16xbatch64 (native batch layout),rhs=1` | - | 0.306 ± 0.026 | 0.552 ± 0.002 | 0.334 ± 0.012 | 0.338 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch1024 (native batch layout),rhs=1` | - | 0.462 ± 0.006 | 0.778 ± 0.006 | 0.238 ± 0.003 | 0.699 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | - | 0.088 ± 0.002 | 0.054 ± 0.000 | 0.078 ± 0.002 | 0.028 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch256 (native batch layout),rhs=1` | - | 0.177 ± 0.002 | 0.229 ± 0.003 | 0.124 ± 0.006 | 0.192 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | - | 0.107 ± 0.003 | 0.090 ± 0.001 | 0.087 ± 0.001 | 0.068 ± 0.001 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch1024 (native batch layout),rhs=1` | - | 0.622 ± 0.030 | 1.146 ± 0.008 | 0.379 ± 0.010 | 0.986 ± 0.009 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | - | 0.092 ± 0.003 | 0.060 ± 0.000 | 0.081 ± 0.002 | 0.043 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch256 (native batch layout),rhs=1` | - | 0.216 ± 0.007 | 0.319 ± 0.002 | 0.153 ± 0.005 | 0.263 ± 0.006 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | - | 0.118 ± 0.003 | 0.112 ± 0.001 | 0.093 ± 0.001 | 0.089 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch1024 (native batch layout),rhs=1` | - | 1.391 ± 0.012 | 3.103 ± 0.011 | 1.344 ± 0.069 | 1.860 ± 0.030 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch16 (native batch layout),rhs=1` | - | 0.105 ± 0.003 | 0.090 ± 0.001 | 0.096 ± 0.001 | 0.057 ± 0.002 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch256 (native batch layout),rhs=1` | - | 0.424 ± 0.016 | 0.802 ± 0.003 | 0.396 ± 0.014 | 0.482 ± 0.005 | - | - |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `8x8xbatch64 (native batch layout),rhs=1` | - | 0.168 ± 0.002 | 0.232 ± 0.001 | 0.155 ± 0.003 | 0.144 ± 0.005 | - | - |
| large | `eigh` | f64 | 1 | `1024x1024` | - | 283.904 ± 1.112 | 282.394 ± 0.670 | 381.620 ± 1.447 | 323.921 ± 1.390 | - | - |
| large | `eigh` | f64 | 1 | `128x128` | - | 2.212 ± 0.020 | 2.200 ± 0.008 | 1.854 ± 0.038 | 1.921 ± 0.031 | - | - |
| large | `eigh` | f64 | 1 | `256x256` | - | 9.500 ± 0.031 | 9.558 ± 0.053 | 9.848 ± 0.038 | 9.289 ± 0.057 | - | - |
| large | `eigh` | f64 | 1 | `512x512` | - | 48.860 ± 0.438 | 48.769 ± 0.475 | 58.935 ± 0.377 | 53.600 ± 0.298 | - | - |
| large | `eigh` | f64 | 1 | `64x64` | - | 0.441 ± 0.001 | 0.444 ± 0.004 | 0.403 ± 0.008 | 0.472 ± 0.006 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `1024x1024` | - | 998.446 ± 0.895 | 498.099 ± 1.639 | 635.665 ± 8.507 | 484.899 ± 1.870 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 31.558 ± 0.195 | 16.283 ± 0.096 | 14.149 ± 0.114 | 11.705 ± 0.047 | - | - |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 165.664 ± 0.736 | 84.527 ± 1.229 | 95.196 ± 0.632 | 76.998 ± 0.666 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `1024x1024` | - | 713.765 ± 1.320 | 496.957 ± 2.687 | 457.023 ± 3.657 | 484.523 ± 3.690 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 21.619 ± 0.211 | 16.290 ± 0.045 | 11.496 ± 0.143 | 11.720 ± 0.045 | - | - |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 116.931 ± 0.690 | 84.397 ± 0.702 | 73.052 ± 0.638 | 75.788 ± 1.533 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `1024x1024` | - | 770.984 ± 1.425 | 350.794 ± 1.465 | 776.907 ± 3.882 | 353.168 ± 3.180 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 14.496 ± 0.136 | 6.587 ± 0.019 | 14.012 ± 0.187 | 6.673 ± 0.018 | - | - |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 98.386 ± 0.211 | 46.414 ± 0.251 | 111.334 ± 0.496 | 47.568 ± 0.392 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `1024x1024` | - | 752.504 ± 1.521 | 359.676 ± 1.230 | 407.921 ± 1.919 | 351.747 ± 1.410 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 13.154 ± 0.027 | 6.892 ± 0.009 | 7.477 ± 0.051 | 6.456 ± 0.015 | - | - |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 95.808 ± 0.314 | 49.293 ± 0.103 | 56.674 ± 0.442 | 46.968 ± 0.236 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `1024x1024` | - | 71.015 ± 0.239 | 69.427 ± 0.280 | 82.967 ± 0.519 | 78.875 ± 1.427 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `128x128` | - | 0.165 ± 0.002 | 0.143 ± 0.001 | 0.208 ± 0.004 | 0.218 ± 0.004 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `256x256` | - | 1.149 ± 0.002 | 1.086 ± 0.001 | 1.359 ± 0.033 | 1.406 ± 0.021 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `512x512` | - | 8.769 ± 0.029 | 8.549 ± 0.013 | 11.481 ± 0.097 | 10.399 ± 0.139 | - | - |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | - | 0.036 ± 0.001 | 0.024 ± 0.000 | 0.039 ± 0.001 | 0.065 ± 0.002 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `1024x1024` | - | 216.680 ± 1.002 | 208.787 ± 0.516 | 251.887 ± 1.279 | 234.550 ± 1.638 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `128x128` | - | 0.566 ± 0.006 | 0.439 ± 0.004 | 0.617 ± 0.021 | 0.537 ± 0.006 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `256x256` | - | 3.600 ± 0.011 | 3.307 ± 0.012 | 3.943 ± 0.083 | 3.845 ± 0.036 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `512x512` | - | 27.358 ± 0.093 | 25.920 ± 0.124 | 32.502 ± 0.172 | 29.905 ± 0.196 | - | - |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | - | 0.140 ± 0.002 | 0.070 ± 0.000 | 0.133 ± 0.004 | 0.096 ± 0.002 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `1024x1024` | - | 1120.745 ± 4.387 | 538.852 ± 0.795 | 533.443 ± 6.105 | 493.573 ± 0.744 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 22.205 ± 0.184 | 10.506 ± 0.012 | 10.610 ± 0.106 | 9.254 ± 0.097 | - | - |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 148.927 ± 0.590 | 73.706 ± 0.229 | 84.505 ± 1.535 | 67.579 ± 0.299 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `1024x1024` | - | 966.325 ± 1.594 | 539.772 ± 2.344 | 533.453 ± 5.853 | 495.749 ± 5.856 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 17.616 ± 0.177 | 10.556 ± 0.085 | 10.651 ± 0.100 | 9.216 ± 0.029 | - | - |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 127.674 ± 0.613 | 73.208 ± 0.204 | 84.870 ± 0.435 | 68.296 ± 0.500 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `1024x1024,rhs=1` | - | 30.348 ± 0.131 | 85.325 ± 0.236 | 49.649 ± 0.399 | 40.718 ± 0.297 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `128x128,rhs=1` | - | 0.246 ± 0.009 | 0.393 ± 0.002 | 0.270 ± 0.008 | 0.234 ± 0.005 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `256x256,rhs=1` | - | 0.826 ± 0.006 | 1.973 ± 0.004 | 1.225 ± 0.042 | 1.049 ± 0.015 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `512x512,rhs=1` | - | 4.453 ± 0.044 | 11.903 ± 0.060 | 7.301 ± 0.113 | 6.988 ± 0.070 | - | - |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | - | 0.128 ± 0.004 | 0.124 ± 0.001 | 0.117 ± 0.003 | 0.085 ± 0.005 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `1024x1024,rhs=1` | - | 85.857 ± 0.602 | 56.850 ± 0.129 | 52.967 ± 4.595 | 41.283 ± 0.292 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 2.197 ± 0.016 | 1.313 ± 0.006 | 1.402 ± 0.047 | 0.952 ± 0.008 | - | - |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 12.828 ± 0.184 | 7.984 ± 0.069 | 7.759 ± 0.058 | 6.564 ± 0.034 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `1024x1024,rhs=1` | - | 57.428 ± 0.285 | 56.331 ± 0.237 | 90.104 ± 4.581 | 40.137 ± 0.202 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.438 ± 0.036 | 1.309 ± 0.008 | 2.080 ± 0.043 | 0.966 ± 0.008 | - | - |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 8.447 ± 0.098 | 7.890 ± 0.043 | 13.135 ± 0.085 | 6.621 ± 0.017 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `1024x1024` | - | 841.595 ± 15.078 | 1116.619 ± 2.149 | 959.904 ± 6.229 | 913.339 ± 4.196 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `128x128` | - | 3.703 ± 0.024 | 5.055 ± 0.017 | 4.049 ± 0.031 | 3.620 ± 0.037 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `256x256` | - | 20.532 ± 0.211 | 28.774 ± 0.103 | 21.568 ± 0.215 | 20.805 ± 0.120 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `512x512` | - | 122.735 ± 0.654 | 168.048 ± 1.307 | 159.250 ± 0.632 | 144.624 ± 0.943 | - | - |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | - | 0.961 ± 0.005 | 1.230 ± 0.015 | 0.888 ± 0.019 | 0.800 ± 0.011 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `1024x1024` | - | 2230.149 ± 5.820 | 1089.630 ± 2.375 | 1153.383 ± 5.588 | 911.048 ± 5.199 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 57.237 ± 0.717 | 28.800 ± 0.133 | 25.790 ± 0.070 | 20.596 ± 0.076 | - | - |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 331.189 ± 2.112 | 167.695 ± 0.798 | 189.945 ± 1.542 | 145.171 ± 0.922 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `1024x1024` | - | 1515.028 ± 3.219 | 1116.067 ± 4.009 | 871.767 ± 4.545 | 912.195 ± 3.500 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 38.637 ± 0.140 | 28.668 ± 0.111 | 21.684 ± 0.279 | 20.638 ± 0.076 | - | - |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 227.172 ± 0.494 | 168.524 ± 1.756 | 164.125 ± 3.678 | 144.409 ± 0.973 | - | - |
| large | `matmul` | f64 | 1 | `1024x1024` | - | 69.341 ± 0.395 | 69.275 ± 0.245 | 78.103 ± 7.084 | 75.770 ± 0.626 | - | - |
| large | `matmul` | f64 | 1 | `128x128` | - | 0.141 ± 0.000 | 0.139 ± 0.000 | 0.234 ± 0.002 | 0.213 ± 0.005 | - | - |
| large | `matmul` | f64 | 1 | `256x256` | - | 1.083 ± 0.005 | 1.079 ± 0.005 | 1.342 ± 0.023 | 1.515 ± 0.052 | - | - |
| large | `matmul` | f64 | 1 | `512x512` | - | 8.495 ± 0.027 | 8.558 ± 0.067 | 10.546 ± 0.069 | 9.994 ± 0.131 | - | - |
| large | `matmul_rect` | f64 | 1 | `1024x256 * 256x1024` | - | 17.082 ± 0.016 | 17.082 ± 0.020 | 19.492 ± 0.123 | 19.115 ± 0.209 | - | - |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | - | 4.276 ± 0.005 | 4.275 ± 0.005 | 5.346 ± 0.087 | 5.211 ± 0.051 | - | - |
| large | `qr` | f64 | 1 | `1024x1024` | - | 136.027 ± 0.861 | 135.521 ± 0.418 | 208.087 ± 4.583 | 175.920 ± 0.387 | - | - |
| large | `qr` | f64 | 1 | `128x128` | - | 0.547 ± 0.008 | 0.548 ± 0.002 | 0.834 ± 0.017 | 0.774 ± 0.010 | - | - |
| large | `qr` | f64 | 1 | `256x256` | - | 2.994 ± 0.007 | 2.993 ± 0.015 | 4.878 ± 0.027 | 3.993 ± 0.087 | - | - |
| large | `qr` | f64 | 1 | `512x512` | - | 18.991 ± 0.091 | 18.837 ± 0.105 | 33.894 ± 0.090 | 27.993 ± 0.126 | - | - |
| large | `qr` | f64 | 1 | `64x64` | - | 0.133 ± 0.000 | 0.132 ± 0.001 | 0.136 ± 0.003 | 0.127 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=1` | - | 26.895 ± 0.108 | 27.189 ± 0.150 | 47.868 ± 0.409 | 40.376 ± 0.484 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=16` | - | 28.133 ± 0.154 | 28.367 ± 0.153 | 49.865 ± 0.379 | 41.900 ± 0.219 | - | - |
| large | `solve` | f64 | 1 | `1024x1024,rhs=64` | - | 32.330 ± 0.201 | 32.294 ± 0.429 | 54.796 ± 0.439 | 46.679 ± 0.111 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=1` | - | 0.114 ± 0.001 | 0.111 ± 0.000 | 0.179 ± 0.004 | 0.195 ± 0.008 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=16` | - | 0.141 ± 0.001 | 0.129 ± 0.002 | 0.232 ± 0.005 | 0.285 ± 0.009 | - | - |
| large | `solve` | f64 | 1 | `128x128,rhs=64` | - | 0.242 ± 0.000 | 0.195 ± 0.001 | 0.384 ± 0.008 | 0.459 ± 0.013 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=1` | - | 0.631 ± 0.007 | 0.618 ± 0.011 | 1.030 ± 0.017 | 0.960 ± 0.015 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=16` | - | 0.716 ± 0.002 | 0.685 ± 0.011 | 1.171 ± 0.013 | 1.149 ± 0.025 | - | - |
| large | `solve` | f64 | 1 | `256x256,rhs=64` | - | 1.047 ± 0.004 | 0.944 ± 0.017 | 1.635 ± 0.036 | 1.630 ± 0.026 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=1` | - | 3.904 ± 0.031 | 3.817 ± 0.067 | 6.870 ± 0.035 | 6.738 ± 0.033 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=16` | - | 4.209 ± 0.049 | 4.108 ± 0.044 | 7.450 ± 0.155 | 7.194 ± 0.043 | - | - |
| large | `solve` | f64 | 1 | `512x512,rhs=64` | - | 5.479 ± 0.024 | 5.213 ± 0.049 | 8.804 ± 0.140 | 8.692 ± 0.061 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | - | 0.032 ± 0.000 | 0.032 ± 0.000 | 0.055 ± 0.001 | 0.060 ± 0.003 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | - | 0.042 ± 0.000 | 0.036 ± 0.000 | 0.073 ± 0.008 | 0.094 ± 0.005 | - | - |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | - | 0.080 ± 0.000 | 0.054 ± 0.000 | 0.123 ± 0.022 | 0.181 ± 0.007 | - | - |
| large | `svd` | f64 | 1 | `1024x1024` | - | 686.956 ± 2.538 | 662.119 ± 3.544 | 795.028 ± 4.841 | 724.240 ± 2.501 | - | - |
| large | `svd` | f64 | 1 | `128x128` | - | 3.290 ± 0.014 | 3.290 ± 0.009 | 3.576 ± 0.046 | 3.319 ± 0.043 | - | - |
| large | `svd` | f64 | 1 | `256x256` | - | 18.033 ± 0.143 | 17.979 ± 0.096 | 19.567 ± 0.080 | 18.269 ± 0.098 | - | - |
| large | `svd` | f64 | 1 | `512x512` | - | 103.985 ± 0.344 | 104.617 ± 0.989 | 125.847 ± 0.583 | 122.067 ± 0.573 | - | - |
| large | `svd` | f64 | 1 | `64x64` | - | 0.826 ± 0.002 | 0.824 ± 0.003 | 0.765 ± 0.028 | 0.740 ± 0.010 | - | - |
| small | `eigh` | f64 | 1 | `16x16` | - | 0.036 ± 0.000 | 0.035 ± 0.000 | 0.034 ± 0.001 | 0.045 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `32x32` | - | 0.121 ± 0.000 | 0.119 ± 0.000 | 0.107 ± 0.001 | 0.134 ± 0.002 | - | - |
| small | `eigh` | f64 | 1 | `4x4` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.000 | - | - |
| small | `eigh` | f64 | 1 | `8x8` | - | 0.014 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `16x16` | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.169 ± 0.003 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.146 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.021 ± 0.001 | 0.176 ± 0.005 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.146 ± 0.002 | - | - |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.164 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `16x16` | - | 0.351 ± 0.004 | 0.082 ± 0.001 | 0.158 ± 0.009 | 0.052 ± 0.001 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.251 ± 0.004 | 0.028 ± 0.000 | 0.122 ± 0.002 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `32x32` | - | 0.630 ± 0.016 | 0.235 ± 0.001 | 0.251 ± 0.005 | 0.141 ± 0.002 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.260 ± 0.005 | 0.032 ± 0.000 | 0.124 ± 0.046 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.280 ± 0.006 | 0.043 ± 0.000 | 0.131 ± 0.002 | 0.024 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `16x16` | - | 0.136 ± 0.005 | 0.081 ± 0.000 | 0.135 ± 0.002 | 0.052 ± 0.001 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.073 ± 0.001 | 0.027 ± 0.001 | 0.104 ± 0.003 | 0.012 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `32x32` | - | 0.318 ± 0.003 | 0.236 ± 0.001 | 0.221 ± 0.006 | 0.142 ± 0.002 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.077 ± 0.003 | 0.031 ± 0.000 | 0.108 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.090 ± 0.002 | 0.043 ± 0.001 | 0.113 ± 0.003 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `16x16` | - | 0.957 ± 0.015 | 0.046 ± 0.000 | 0.219 ± 0.006 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.917 ± 0.017 | 0.037 ± 0.000 | 0.197 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `32x32` | - | 1.010 ± 0.033 | 0.073 ± 0.001 | 0.274 ± 0.003 | 0.051 ± 0.001 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.912 ± 0.018 | 0.037 ± 0.006 | 0.195 ± 0.006 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.934 ± 0.011 | 0.040 ± 0.000 | 0.198 ± 0.005 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `16x16` | - | 0.196 ± 0.002 | 0.044 ± 0.000 | 0.173 ± 0.007 | 0.026 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.178 ± 0.002 | 0.034 ± 0.001 | 0.158 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `32x32` | - | 0.247 ± 0.006 | 0.072 ± 0.000 | 0.215 ± 0.008 | 0.051 ± 0.001 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.177 ± 0.002 | 0.034 ± 0.001 | 0.160 ± 0.006 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.182 ± 0.003 | 0.036 ± 0.000 | 0.160 ± 0.005 | 0.021 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `16x16` | - | 0.061 ± 0.001 | 0.015 ± 0.000 | 0.047 ± 0.000 | 0.024 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | - | 0.057 ± 0.001 | 0.014 ± 0.000 | 0.042 ± 0.001 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `32x32` | - | 0.070 ± 0.001 | 0.022 ± 0.000 | 0.060 ± 0.001 | 0.039 ± 0.002 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | - | 0.058 ± 0.001 | 0.014 ± 0.000 | 0.043 ± 0.001 | 0.014 ± 0.001 | - | - |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | - | 0.058 ± 0.001 | 0.015 ± 0.000 | 0.044 ± 0.001 | 0.024 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `16x16` | - | 1.378 ± 0.046 | 0.069 ± 0.001 | 0.157 ± 0.011 | 0.030 ± 0.001 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 1.237 ± 0.036 | 0.046 ± 0.000 | 0.139 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `32x32` | - | 1.451 ± 0.030 | 0.133 ± 0.004 | 0.201 ± 0.007 | 0.062 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 1.293 ± 0.035 | 0.047 ± 0.000 | 0.139 ± 0.004 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 1.308 ± 0.043 | 0.052 ± 0.000 | 0.143 ± 0.003 | 0.021 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `16x16` | - | 0.189 ± 0.005 | 0.067 ± 0.001 | 0.164 ± 0.007 | 0.030 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.156 ± 0.003 | 0.045 ± 0.000 | 0.145 ± 0.003 | 0.013 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `32x32` | - | 0.271 ± 0.005 | 0.130 ± 0.001 | 0.204 ± 0.004 | 0.061 ± 0.001 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.160 ± 0.004 | 0.046 ± 0.000 | 0.147 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.165 ± 0.003 | 0.050 ± 0.000 | 0.149 ± 0.004 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `16x16,rhs=1` | - | 0.082 ± 0.001 | 0.050 ± 0.000 | 0.071 ± 0.000 | 0.022 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | - | 0.076 ± 0.002 | 0.041 ± 0.000 | 0.065 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `32x32,rhs=1` | - | 0.093 ± 0.002 | 0.066 ± 0.001 | 0.083 ± 0.001 | 0.041 ± 0.007 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | - | 0.080 ± 0.001 | 0.041 ± 0.000 | 0.068 ± 0.002 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | - | 0.080 ± 0.002 | 0.045 ± 0.000 | 0.068 ± 0.001 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `16x16,rhs=1` | - | 0.194 ± 0.004 | 0.034 ± 0.000 | 0.337 ± 0.007 | 0.018 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.181 ± 0.006 | 0.028 ± 0.000 | 0.321 ± 0.002 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `32x32,rhs=1` | - | 0.213 ± 0.005 | 0.044 ± 0.000 | 0.352 ± 0.004 | 0.029 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.182 ± 0.004 | 0.029 ± 0.000 | 0.325 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.184 ± 0.003 | 0.030 ± 0.000 | 0.328 ± 0.003 | 0.015 ± 0.001 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `16x16,rhs=1` | - | 0.091 ± 0.001 | 0.033 ± 0.000 | 0.147 ± 0.003 | 0.020 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.085 ± 0.001 | 0.028 ± 0.000 | 0.140 ± 0.003 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `32x32,rhs=1` | - | 0.103 ± 0.001 | 0.044 ± 0.001 | 0.170 ± 0.010 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.085 ± 0.001 | 0.028 ± 0.000 | 0.140 ± 0.002 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.088 ± 0.002 | 0.030 ± 0.000 | 0.140 ± 0.001 | 0.016 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `16x16` | - | 0.126 ± 0.005 | 0.097 ± 0.000 | 0.109 ± 0.002 | 0.079 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | - | 0.063 ± 0.002 | 0.020 ± 0.000 | 0.048 ± 0.000 | 0.015 ± 0.000 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `32x32` | - | 0.284 ± 0.006 | 0.309 ± 0.001 | 0.258 ± 0.002 | 0.213 ± 0.004 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | - | 0.069 ± 0.002 | 0.026 ± 0.000 | 0.054 ± 0.000 | 0.016 ± 0.001 | - | - |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | - | 0.078 ± 0.002 | 0.041 ± 0.000 | 0.068 ± 0.000 | 0.039 ± 0.008 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `16x16` | - | 0.396 ± 0.003 | 0.098 ± 0.000 | 0.258 ± 0.003 | 0.077 ± 0.001 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.212 ± 0.003 | 0.020 ± 0.000 | 0.182 ± 0.005 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `32x32` | - | 0.869 ± 0.007 | 0.309 ± 0.001 | 0.442 ± 0.017 | 0.209 ± 0.002 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.236 ± 0.019 | 0.026 ± 0.000 | 0.191 ± 0.006 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.262 ± 0.006 | 0.042 ± 0.000 | 0.211 ± 0.007 | 0.033 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `16x16` | - | 0.187 ± 0.003 | 0.097 ± 0.000 | 0.178 ± 0.006 | 0.078 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.075 ± 0.001 | 0.020 ± 0.000 | 0.115 ± 0.003 | 0.014 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `32x32` | - | 0.497 ± 0.013 | 0.310 ± 0.002 | 0.336 ± 0.012 | 0.209 ± 0.001 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.084 ± 0.003 | 0.025 ± 0.000 | 0.121 ± 0.004 | 0.017 ± 0.000 | - | - |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.104 ± 0.003 | 0.041 ± 0.000 | 0.136 ± 0.005 | 0.033 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `16x16` | - | 0.005 ± 0.002 | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.020 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `2x2` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `32x32` | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.021 ± 0.001 | - | - |
| small | `matmul` | f64 | 1 | `4x4` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.002 ± 0.000 | 0.009 ± 0.000 | - | - |
| small | `matmul` | f64 | 1 | `64x64` | - | 0.021 ± 0.000 | 0.022 ± 0.000 | 0.043 ± 0.001 | 0.051 ± 0.002 | - | - |
| small | `matmul` | f64 | 1 | `8x8` | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.003 ± 0.000 | 0.019 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `16x16` | - | 0.015 ± 0.000 | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.019 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `2x2` | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `32x32` | - | 0.039 ± 0.000 | 0.039 ± 0.001 | 0.032 ± 0.001 | 0.047 ± 0.001 | - | - |
| small | `qr` | f64 | 1 | `4x4` | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.000 | - | - |
| small | `qr` | f64 | 1 | `8x8` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=1` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.020 ± 0.001 | 0.013 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `16x16,rhs=4` | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=1` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.028 ± 0.001 | 0.028 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `32x32,rhs=4` | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.029 ± 0.000 | 0.030 ± 0.008 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.010 ± 0.000 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.011 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `16x16` | - | 0.061 ± 0.000 | 0.060 ± 0.000 | 0.065 ± 0.000 | 0.072 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `2x2` | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `32x32` | - | 0.212 ± 0.002 | 0.210 ± 0.001 | 0.208 ± 0.001 | 0.213 ± 0.003 | - | - |
| small | `svd` | f64 | 1 | `4x4` | - | 0.011 ± 0.000 | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.018 ± 0.001 | - | - |
| small | `svd` | f64 | 1 | `8x8` | - | 0.021 ± 0.000 | 0.020 ± 0.000 | 0.026 ± 0.000 | 0.028 ± 0.001 | - | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-eager` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`16x16xbatch16 (native batch layout)`): `tenferro-trace` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.182 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-eager` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch16 (native batch layout)`): `tenferro-trace` is 22.2x faster than the slowest successful cell (`jax-cpu`, 0.149 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-eager` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`2x2xbatch64 (native batch layout)`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.166 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `pytorch-cpu` is 10.0x faster than the slowest successful cell (`jax-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-eager` is 23.2x faster than the slowest successful cell (`jax-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch16 (native batch layout)`): `tenferro-trace` is 24.0x faster than the slowest successful cell (`jax-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-eager` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`4x4xbatch64 (native batch layout)`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.171 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-eager` is 22.3x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch16 (native batch layout)`): `tenferro-trace` is 22.9x faster than the slowest successful cell (`jax-cpu`, 0.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-eager` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/batched_matmul_ikb_kjb_ijb` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`2x2xbatch1024 (native batch layout)`): `jax-cpu` is 11.7x faster than the slowest successful cell (`tenferro-eager`, 0.529 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch1024 (native batch layout)`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 7.375 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch256 (native batch layout)`): `tenferro-trace` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 1.911 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `batched/grad_sum_batched_matmul_backward` (f64, threads=1, shape=`8x8xbatch64 (native batch layout)`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.563 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-eager` is 34.0x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.169 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-eager` is 32.8x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-eager` is 23.9x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 15.2x faster than the slowest successful cell (`jax-cpu`, 0.176 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-eager` is 32.4x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-eager` is 33.1x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/einsum_ij_jk_ik` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.251 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 18.5x faster than the slowest successful cell (`tenferro-eager`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_eigh_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 0.280 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 36.8x faster than the slowest successful cell (`tenferro-eager`, 0.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 20.8x faster than the slowest successful cell (`tenferro-eager`, 0.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 66.2x faster than the slowest successful cell (`tenferro-eager`, 0.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 25.0x faster than the slowest successful cell (`tenferro-eager`, 0.917 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`32x32`): `jax-cpu` is 19.7x faster than the slowest successful cell (`tenferro-eager`, 1.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 1.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 63.8x faster than the slowest successful cell (`tenferro-eager`, 0.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 24.4x faster than the slowest successful cell (`tenferro-eager`, 0.912 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 45.1x faster than the slowest successful cell (`tenferro-eager`, 0.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`tenferro-eager`, 0.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.7x faster than the slowest successful cell (`tenferro-eager`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_lu_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 0.177 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`16x16`): `jax-cpu` is 45.6x faster than the slowest successful cell (`tenferro-eager`, 1.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`16x16`): `tenferro-trace` is 20.0x faster than the slowest successful cell (`tenferro-eager`, 1.378 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 91.9x faster than the slowest successful cell (`tenferro-eager`, 1.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 26.9x faster than the slowest successful cell (`tenferro-eager`, 1.237 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`32x32`): `jax-cpu` is 23.5x faster than the slowest successful cell (`tenferro-eager`, 1.451 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`32x32`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 1.451 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 89.6x faster than the slowest successful cell (`tenferro-eager`, 1.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`4x4`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`tenferro-eager`, 1.293 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `jax-cpu` is 63.7x faster than the slowest successful cell (`tenferro-eager`, 1.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_jvp` (f64, threads=1, shape=`8x8`): `tenferro-trace` is 25.4x faster than the slowest successful cell (`tenferro-eager`, 1.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 12.2x faster than the slowest successful cell (`tenferro-eager`, 0.156 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_qr_vjp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-eager`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`16x16,rhs=1`): `jax-cpu` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.337 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `jax-cpu` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`2x2,rhs=1`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`pytorch-cpu`, 0.321 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`32x32,rhs=1`): `jax-cpu` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 0.352 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `jax-cpu` is 23.5x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`4x4,rhs=1`): `tenferro-trace` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.325 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `jax-cpu` is 21.9x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_solve_jvp` (f64, threads=1, shape=`8x8,rhs=1`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.328 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `jax-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`2x2`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`tenferro-eager`, 0.212 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `small/grad_sum_svd_s_jvp` (f64, threads=1, shape=`4x4`): `jax-cpu` is 13.8x faster than the slowest successful cell (`tenferro-eager`, 0.236 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

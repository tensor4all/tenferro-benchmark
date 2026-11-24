# CPU Benchmark Results

- Target profile: `zen5-cpu`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`
- Benchmark commit: `bb9869533cc2ad586b71cdb49a3e1f85439cd2b2`

Generated from the explicit runs below; each section retains its own provenance and thread settings.

## Threads: 1

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_120608/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_120608/run.yaml`
- Timestamp: `2026-11-24T03:06:08.486161+00:00`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_120608`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_120608/cpu_ops_t1_20261124_120608.csv`

### CPU Benchmark Items

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

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 4

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_131021/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_131021/run.yaml`
- Timestamp: `2026-11-24T04:10:21.867323+00:00`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_131021`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_131021/cpu_ops_t4_20261124_131021.csv`

### CPU Benchmark Items

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
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | withdrawn | 15.260 ± 0.069 | 5.259 ± 0.056 | 10.561 ± 0.918 | - | - |
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
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | withdrawn | 27.297 ± 0.079 | 15.406 ± 0.149 | 19.041 ± 0.279 | - | - |
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

### Withdrawn cells

These cells were measured, but their number does not measure what the column says,
so it is not printed (`benchmarks/withdrawn-cells.yaml`; the run record is unchanged).

- `grad_sum_svd_s_vjp` 4T `256x256` `tenferro-eager` (tenferro-benchmark#517): timed a second forward pass after the VJP (harness defect from the tenferro-rs 0.12 migration, 03f0f80); re-measured single-pass at the same pin: 18.905 ms (between-start SD 0.103, W 0.210) vs 18.817 ± 0.075 at 9a850b06, a 0.47 % difference inside W, so the two revisions are not distinguishable here.
- `grad_sum_eigh_vjp` 4T `256x256` `tenferro-eager` (tenferro-benchmark#517): timed a second forward pass after the VJP (harness defect from the tenferro-rs 0.12 migration, 03f0f80); re-measured single-pass at the same pin: 10.228 ms (between-start SD 0.060, W 0.078) vs 10.416 ± 0.032 at 9a850b06, 1.8 % faster, so the published 1.92x slowdown does not exist.

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

## Threads: 8

Source report: `data/results/zen5-cpu/cpu/einsum/20261124_135202/cpu_ops_report.md`.


- Suite: `cpu/cpu_ops`
- Target profile: `zen5-cpu`
- Suite file: `benchmarks/cpu/einsum.yaml`
- Run metadata: `data/results/zen5-cpu/cpu/einsum/20261124_135202/run.yaml`
- Timestamp: `2026-11-24T04:52:02.812988+00:00`

Latest run: `./scripts/run_all.sh 8`.

This file is generated from one CPU ops run under `data/results/zen5-cpu/cpu/einsum/20261124_135202`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

### CPU Information

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

### Thread Environment

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

### Tenferro CPU BLAS Backend

- tenferro-rs features: `native`

### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/home/shinaoka/work/zen5-refresh-20261124/harness/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`


- CSV: `data/results/zen5-cpu/cpu/einsum/20261124_135202/cpu_ops_t8_20261124_135202.csv`

### CPU Benchmark Items

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

### Cross-Backend Spread Audit

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

### Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.

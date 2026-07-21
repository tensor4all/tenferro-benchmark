# Linux CPU Ops Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Thread runs: `1 4`

Mirrored from `amd-cpu` runs (`BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh <THREADS>`);
regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260721_064813 4:20260721_071519 permutation:20260721_073112`.

This report is generated from sequential CPU runs. Do not compare it with
measurements collected while another CPU benchmark process was running.

## Run Inputs

- Threads 1: timestamp `20260721_064813`, raw run `data/results/linux-cpu/cpu/einsum/20260721_064813`
- Threads 4: timestamp `20260721_071519`, raw run `data/results/linux-cpu/cpu/einsum/20260721_071519`

## Threads: 1

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260721_064813`

Latest run: `./scripts/run_all.sh 1`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260721_064813`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

#### Thread Environment

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 1

- CSV: `data/results/linux-cpu/cpu/einsum/20260721_064813/cpu_ops_t1_20260721_064813.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260721_064813/cpu_ops_t1_20260721_064813.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.236 ± 0.007 | 0.034 ± 0.002 | 0.575 ± 0.020 | 1.836 ± 0.018 |
| batched | `batched_eigh` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.381 ± 0.018 | 0.105 ± 0.007 | 1.971 ± 0.141 | 5.302 ± 0.062 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.298 ± 0.013 | 0.071 ± 0.009 | 0.780 ± 0.196 | 2.132 ± 0.056 |
| batched | `batched_eigh` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.613 ± 0.014 | 0.260 ± 0.063 | 2.500 ± 0.008 | 6.874 ± 0.053 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.205 ± 0.006 | 0.009 ± 0.000 | 0.152 ± 0.011 | 1.763 ± 0.035 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.218 ± 0.015 | 0.023 ± 0.001 | 0.298 ± 0.005 | 1.968 ± 0.050 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.196 ± 0.061 | 0.009 ± 0.000 | 0.298 ± 0.010 | 1.741 ± 0.106 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.220 ± 0.004 | 0.023 ± 0.001 | 0.818 ± 0.038 | 3.039 ± 0.078 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.230 ± 0.026 | 0.037 ± 0.003 | 0.318 ± 0.011 | 1.245 ± 0.021 |
| batched | `batched_qr` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.327 ± 0.008 | 0.083 ± 0.013 | 1.058 ± 0.013 | 2.352 ± 0.057 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.229 ± 0.009 | 0.035 ± 0.000 | 0.420 ± 0.015 | 1.412 ± 0.054 |
| batched | `batched_qr` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.388 ± 0.013 | 0.107 ± 0.001 | 1.448 ± 0.013 | 4.462 ± 0.066 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.279 ± 0.014 | 0.035 ± 0.000 | 0.639 ± 0.090 | 1.612 ± 0.195 |
| batched | `batched_solve` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.386 ± 0.008 | 0.084 ± 0.000 | 2.033 ± 0.016 | 5.497 ± 0.080 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.291 ± 0.003 | 0.037 ± 0.002 | 0.756 ± 0.015 | 2.357 ± 0.058 |
| batched | `batched_solve` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.433 ± 0.039 | 0.090 ± 0.015 | 2.503 ± 0.104 | 6.806 ± 0.293 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.299 ± 0.024 | 0.057 ± 0.011 | 0.364 ± 0.016 | 1.215 ± 0.016 |
| batched | `batched_svd` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.510 ± 0.054 | 0.202 ± 0.007 | 1.177 ± 0.120 | 2.791 ± 0.063 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.428 ± 0.043 | 0.149 ± 0.015 | 0.531 ± 0.010 | 1.648 ± 0.047 |
| batched | `batched_svd` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.992 ± 0.010 | 0.578 ± 0.090 | 1.795 ± 0.036 | 5.131 ± 0.028 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch16 (native batch layout)` | 0.488 ± 0.006 | 0.051 ± 0.001 | 0.369 ± 0.010 | 5.896 ± 1.864 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `2x2xbatch64 (native batch layout)` | 0.645 ± 0.063 | 0.083 ± 0.008 | 0.531 ± 0.011 | 6.361 ± 1.868 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch16 (native batch layout)` | 0.499 ± 0.045 | 0.052 ± 0.002 | 0.535 ± 0.013 | 6.127 ± 1.949 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 1 | `4x4xbatch64 (native batch layout)` | 0.563 ± 0.031 | 0.083 ± 0.007 | 1.131 ± 0.016 | 6.500 ± 2.672 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch16 (native batch layout),rhs=1` | 0.795 ± 0.017 | 0.161 ± 0.016 | 0.825 ± 0.021 | 6.860 ± 1.418 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `2x2xbatch64 (native batch layout),rhs=1` | 0.952 ± 0.049 | 0.301 ± 0.040 | 2.215 ± 0.048 | 8.169 ± 3.672 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch16 (native batch layout),rhs=1` | 0.811 ± 0.049 | 0.179 ± 0.008 | 0.943 ± 0.012 | 7.036 ± 1.566 |
| batched | `grad_sum_batched_solve_backward` | f64 | 1 | `4x4xbatch64 (native batch layout),rhs=1` | 0.904 ± 0.028 | 0.314 ± 0.026 | 2.716 ± 0.137 | 8.353 ± 3.084 |
| large | `eigh` | f64 | 1 | `64x64` | 1.533 ± 0.042 | 0.504 ± 0.088 | 2.123 ± 0.066 | 6.963 ± 0.065 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `256x256` | - | 12.605 ± 0.231 | 11.787 ± 0.346 | 36.093 ± 1.090 |
| large | `grad_sum_eigh_jvp` | f64 | 1 | `512x512` | - | 68.304 ± 0.666 | 70.486 ± 0.474 | 114.402 ± 1.100 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `256x256` | - | 12.110 ± 0.136 | 9.699 ± 0.522 | 34.913 ± 0.708 |
| large | `grad_sum_eigh_vjp` | f64 | 1 | `512x512` | - | 66.794 ± 0.501 | 52.657 ± 0.378 | 116.734 ± 1.174 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `256x256` | - | 7.499 ± 0.576 | 8.961 ± 0.490 | 12.667 ± 1.933 |
| large | `grad_sum_lu_jvp` | f64 | 1 | `512x512` | - | 40.667 ± 2.864 | 67.826 ± 0.357 | 48.103 ± 3.922 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `256x256` | - | 7.352 ± 0.081 | 5.355 ± 0.240 | 15.816 ± 2.108 |
| large | `grad_sum_lu_vjp` | f64 | 1 | `512x512` | - | 40.735 ± 0.097 | 37.992 ± 0.616 | 54.181 ± 3.586 |
| large | `grad_sum_matmul` | f64 | 1 | `64x64` | 0.301 ± 0.033 | 0.045 ± 0.013 | 3.115 ± 0.200 | 5.610 ± 0.807 |
| large | `grad_sum_matmul_backward` | f64 | 1 | `64x64` | 0.688 ± 0.011 | 0.097 ± 0.011 | 3.188 ± 0.299 | 7.990 ± 2.133 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `256x256` | - | 10.119 ± 0.292 | 7.698 ± 0.265 | 18.492 ± 1.084 |
| large | `grad_sum_qr_jvp` | f64 | 1 | `512x512` | - | 57.100 ± 2.187 | 51.111 ± 0.165 | 79.030 ± 12.633 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `256x256` | - | 10.515 ± 0.609 | 7.440 ± 0.266 | 19.445 ± 1.011 |
| large | `grad_sum_qr_vjp` | f64 | 1 | `512x512` | - | 56.909 ± 0.977 | 49.945 ± 0.574 | 101.139 ± 2.883 |
| large | `grad_sum_solve_backward` | f64 | 1 | `64x64,rhs=1` | 1.219 ± 0.081 | 0.184 ± 0.031 | 1.940 ± 0.210 | 7.817 ± 3.707 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `256x256,rhs=1` | - | 1.523 ± 0.117 | 1.675 ± 0.020 | 5.261 ± 0.052 |
| large | `grad_sum_solve_jvp` | f64 | 1 | `512x512,rhs=1` | - | 6.572 ± 0.075 | 5.529 ± 0.159 | 15.572 ± 0.182 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `256x256,rhs=1` | - | 1.160 ± 0.164 | 2.040 ± 0.002 | 5.909 ± 0.104 |
| large | `grad_sum_solve_vjp` | f64 | 1 | `512x512,rhs=1` | - | 5.219 ± 0.244 | 8.875 ± 0.449 | 16.310 ± 0.119 |
| large | `grad_sum_svd_s_backward` | f64 | 1 | `64x64` | 2.555 ± 0.084 | 0.951 ± 0.059 | 2.564 ± 0.127 | 7.862 ± 2.071 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `256x256` | - | 19.233 ± 0.422 | 21.960 ± 0.361 | 62.300 ± 1.051 |
| large | `grad_sum_svd_s_jvp` | f64 | 1 | `512x512` | - | 118.464 ± 1.175 | 141.707 ± 0.607 | 204.113 ± 4.398 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `256x256` | - | 19.074 ± 0.409 | 18.387 ± 0.652 | 60.367 ± 1.556 |
| large | `grad_sum_svd_s_vjp` | f64 | 1 | `512x512` | - | 116.958 ± 1.374 | 114.339 ± 1.285 | 204.836 ± 2.740 |
| large | `matmul` | f64 | 1 | `128x128` | 0.807 ± 0.022 | 0.203 ± 0.003 | 11.425 ± 0.506 | 18.097 ± 0.224 |
| large | `matmul` | f64 | 1 | `256x256` | 3.153 ± 0.297 | 1.185 ± 0.044 | 45.328 ± 0.398 | 68.806 ± 1.489 |
| large | `matmul_rect` | f64 | 1 | `256x1024 * 1024x256` | 8.109 ± 0.045 | 4.372 ± 0.203 | 183.063 ± 3.205 | 180.390 ± 2.402 |
| large | `qr` | f64 | 1 | `64x64` | 0.516 ± 0.040 | 0.118 ± 0.022 | 1.734 ± 0.023 | 5.349 ± 0.049 |
| large | `solve` | f64 | 1 | `64x64,rhs=1` | 0.860 ± 0.023 | 0.059 ± 0.006 | 1.837 ± 0.215 | 5.712 ± 0.037 |
| large | `solve` | f64 | 1 | `64x64,rhs=16` | 0.864 ± 0.044 | 0.066 ± 0.012 | 2.177 ± 0.041 | 6.716 ± 0.067 |
| large | `solve` | f64 | 1 | `64x64,rhs=64` | 0.889 ± 0.015 | 0.126 ± 0.011 | 3.302 ± 0.098 | 8.926 ± 1.627 |
| large | `svd` | f64 | 1 | `64x64` | 1.618 ± 0.024 | 0.828 ± 0.020 | 2.443 ± 0.134 | 7.057 ± 1.443 |
| small | `eigh` | f64 | 1 | `2x2` | 0.284 ± 0.020 | 0.010 ± 0.001 | 0.106 ± 0.034 | 0.421 ± 0.020 |
| small | `eigh` | f64 | 1 | `4x4` | 0.310 ± 0.031 | 0.015 ± 0.000 | 0.113 ± 0.002 | 0.744 ± 0.034 |
| small | `eigh` | f64 | 1 | `8x8` | 0.261 ± 0.005 | 0.019 ± 0.000 | 0.140 ± 0.003 | 0.793 ± 0.016 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `2x2` | 0.233 ± 0.021 | 0.006 ± 0.000 | 0.112 ± 0.003 | 0.662 ± 0.068 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `4x4` | 0.248 ± 0.008 | 0.006 ± 0.000 | 0.127 ± 0.017 | 1.511 ± 0.030 |
| small | `einsum_ij_jk_ik` | f64 | 1 | `8x8` | 0.231 ± 0.006 | 0.005 ± 0.000 | 0.166 ± 0.012 | 1.811 ± 0.020 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `2x2` | - | 0.117 ± 0.002 | 0.326 ± 0.009 | 1.543 ± 0.028 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `4x4` | - | 0.112 ± 0.003 | 0.329 ± 0.012 | 1.570 ± 0.040 |
| small | `grad_sum_eigh_jvp` | f64 | 1 | `8x8` | - | 0.112 ± 0.007 | 0.347 ± 0.026 | 1.763 ± 0.029 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `2x2` | - | 0.128 ± 0.004 | 0.388 ± 0.123 | 3.694 ± 0.151 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `4x4` | - | 0.123 ± 0.008 | 0.324 ± 0.011 | 3.777 ± 0.131 |
| small | `grad_sum_eigh_vjp` | f64 | 1 | `8x8` | - | 0.121 ± 0.007 | 0.336 ± 0.011 | 4.869 ± 0.206 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `2x2` | - | 0.123 ± 0.003 | 0.460 ± 0.035 | 1.873 ± 0.013 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `4x4` | - | 0.110 ± 0.004 | 0.503 ± 0.069 | 2.105 ± 0.056 |
| small | `grad_sum_lu_jvp` | f64 | 1 | `8x8` | - | 0.103 ± 0.003 | 0.488 ± 0.091 | 3.112 ± 0.031 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `2x2` | - | 0.130 ± 0.010 | 0.418 ± 0.057 | 5.642 ± 0.313 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `4x4` | - | 0.115 ± 0.003 | 0.423 ± 0.047 | 5.759 ± 0.422 |
| small | `grad_sum_lu_vjp` | f64 | 1 | `8x8` | - | 0.106 ± 0.006 | 0.419 ± 0.038 | 6.009 ± 2.076 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `2x2` | 0.704 ± 0.054 | 0.050 ± 0.002 | 0.225 ± 0.053 | 5.270 ± 0.903 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `4x4` | 0.723 ± 0.022 | 0.055 ± 0.000 | 0.264 ± 0.036 | 5.217 ± 0.854 |
| small | `grad_sum_matmul_backward` | f64 | 1 | `8x8` | 0.623 ± 0.033 | 0.046 ± 0.001 | 0.272 ± 0.017 | 5.431 ± 0.999 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `2x2` | - | 0.281 ± 0.014 | 0.362 ± 0.008 | 2.640 ± 0.527 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `4x4` | - | 0.270 ± 0.012 | 0.355 ± 0.010 | 2.852 ± 0.078 |
| small | `grad_sum_qr_jvp` | f64 | 1 | `8x8` | - | 0.260 ± 0.047 | 0.358 ± 0.008 | 2.785 ± 0.038 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `2x2` | - | 0.665 ± 0.041 | 0.394 ± 0.021 | 6.946 ± 1.357 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `4x4` | - | 0.603 ± 0.037 | 0.394 ± 0.017 | 6.591 ± 1.169 |
| small | `grad_sum_qr_vjp` | f64 | 1 | `8x8` | - | 0.514 ± 0.016 | 0.404 ± 0.014 | 6.821 ± 1.251 |
| small | `grad_sum_solve_backward` | f64 | 1 | `2x2,rhs=1` | 1.022 ± 0.041 | 0.151 ± 0.003 | 0.317 ± 0.008 | 5.873 ± 0.639 |
| small | `grad_sum_solve_backward` | f64 | 1 | `4x4,rhs=1` | 0.935 ± 0.018 | 0.157 ± 0.001 | 0.335 ± 0.026 | 5.880 ± 0.732 |
| small | `grad_sum_solve_backward` | f64 | 1 | `8x8,rhs=1` | 0.837 ± 0.010 | 0.133 ± 0.002 | 0.366 ± 0.025 | 6.041 ± 0.736 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `2x2,rhs=1` | - | 0.077 ± 0.002 | 0.677 ± 0.009 | 2.360 ± 0.145 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `4x4,rhs=1` | - | 0.068 ± 0.001 | 0.682 ± 0.012 | 2.377 ± 0.081 |
| small | `grad_sum_solve_jvp` | f64 | 1 | `8x8,rhs=1` | - | 0.063 ± 0.003 | 0.682 ± 0.006 | 2.193 ± 0.016 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `2x2,rhs=1` | - | 0.115 ± 0.005 | 0.443 ± 0.009 | 5.121 ± 0.426 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `4x4,rhs=1` | - | 0.102 ± 0.003 | 0.448 ± 0.008 | 5.171 ± 0.211 |
| small | `grad_sum_solve_vjp` | f64 | 1 | `8x8,rhs=1` | - | 0.088 ± 0.008 | 0.452 ± 0.013 | 5.161 ± 0.122 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `2x2` | 1.676 ± 0.028 | 0.089 ± 0.002 | 0.230 ± 0.008 | 5.492 ± 0.599 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `4x4` | 1.677 ± 0.058 | 0.100 ± 0.005 | 0.248 ± 0.009 | 5.575 ± 0.649 |
| small | `grad_sum_svd_s_backward` | f64 | 1 | `8x8` | 1.441 ± 0.054 | 0.105 ± 0.004 | 0.341 ± 0.082 | 5.754 ± 0.700 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `2x2` | - | 0.073 ± 0.003 | 0.427 ± 0.009 | 1.867 ± 0.058 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `4x4` | - | 0.088 ± 0.004 | 0.437 ± 0.011 | 1.926 ± 0.044 |
| small | `grad_sum_svd_s_jvp` | f64 | 1 | `8x8` | - | 0.088 ± 0.003 | 0.453 ± 0.009 | 1.851 ± 0.054 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `2x2` | - | 0.086 ± 0.006 | 0.355 ± 0.019 | 5.061 ± 0.366 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `4x4` | - | 0.102 ± 0.002 | 0.348 ± 0.010 | 5.093 ± 0.279 |
| small | `grad_sum_svd_s_vjp` | f64 | 1 | `8x8` | - | 0.101 ± 0.003 | 0.374 ± 0.016 | 5.159 ± 0.293 |
| small | `matmul` | f64 | 1 | `2x2` | 0.238 ± 0.017 | 0.006 ± 0.000 | 0.079 ± 0.005 | 0.414 ± 0.060 |
| small | `matmul` | f64 | 1 | `4x4` | 0.240 ± 0.009 | 0.006 ± 0.000 | 0.089 ± 0.006 | 0.804 ± 0.031 |
| small | `matmul` | f64 | 1 | `8x8` | 0.226 ± 0.005 | 0.005 ± 0.000 | 0.129 ± 0.005 | 1.180 ± 0.034 |
| small | `qr` | f64 | 1 | `2x2` | 0.260 ± 0.029 | 0.010 ± 0.000 | 0.071 ± 0.003 | 0.277 ± 0.009 |
| small | `qr` | f64 | 1 | `4x4` | 0.259 ± 0.004 | 0.010 ± 0.001 | 0.077 ± 0.001 | 0.633 ± 0.023 |
| small | `qr` | f64 | 1 | `8x8` | 0.238 ± 0.012 | 0.010 ± 0.001 | 0.099 ± 0.002 | 0.699 ± 0.017 |
| small | `solve` | f64 | 1 | `2x2,rhs=1` | 0.358 ± 0.058 | 0.022 ± 0.001 | 0.183 ± 0.046 | 0.664 ± 0.022 |
| small | `solve` | f64 | 1 | `2x2,rhs=4` | 0.368 ± 0.035 | 0.025 ± 0.001 | 0.155 ± 0.002 | 0.605 ± 0.021 |
| small | `solve` | f64 | 1 | `4x4,rhs=1` | 0.328 ± 0.028 | 0.022 ± 0.001 | 0.162 ± 0.002 | 0.795 ± 0.056 |
| small | `solve` | f64 | 1 | `4x4,rhs=4` | 0.355 ± 0.016 | 0.025 ± 0.001 | 0.167 ± 0.004 | 0.801 ± 0.013 |
| small | `solve` | f64 | 1 | `8x8,rhs=1` | 0.307 ± 0.050 | 0.019 ± 0.001 | 0.186 ± 0.009 | 0.870 ± 0.009 |
| small | `solve` | f64 | 1 | `8x8,rhs=4` | 0.319 ± 0.044 | 0.021 ± 0.001 | 0.195 ± 0.010 | 0.872 ± 0.008 |
| small | `svd` | f64 | 1 | `2x2` | 0.305 ± 0.033 | 0.014 ± 0.000 | 0.092 ± 0.003 | 0.255 ± 0.009 |
| small | `svd` | f64 | 1 | `4x4` | 0.295 ± 0.024 | 0.023 ± 0.001 | 0.104 ± 0.002 | 0.646 ± 0.024 |
| small | `svd` | f64 | 1 | `8x8` | 0.319 ± 0.010 | 0.033 ± 0.001 | 0.141 ± 0.004 | 0.654 ± 0.022 |

## Threads: 4

### CPU Benchmark Results

- Suite: `cpu/cpu_ops`
- Target profile: `linux-cpu`
- Timestamp: `20260721_071519`

Latest run: `./scripts/run_all.sh 4`.

This file is generated from one CPU ops run under `data/results/linux-cpu/cpu/einsum/20260721_071519`.

- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

#### CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

#### Thread Environment

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

#### Tenferro CPU BLAS Backend

- tenferro-rs features: `system-mkl`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `mkl`
- BLAS version: `2026.0.0`
- BLAS root: `/opt/intel/oneapi/mkl/latest`
- BLAS library: `/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so`

#### Python Backend Providers

- PyTorch: BLAS provider `mkl`, version `2.12.0+cpu`, BLAS_INFO `mkl`, LAPACK_INFO `mkl`
  - linked BLAS/LAPACK libs: `/workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `none_detected`

#### Threads: 4

- CSV: `data/results/linux-cpu/cpu/einsum/20260721_071519/cpu_ops_t4_20260721_071519.csv`
- Source table: `data/results/linux-cpu/cpu/einsum/20260721_071519/cpu_ops_t4_20260721_071519.md`

#### CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.428 ± 0.039 | 0.043 ± 0.000 | 0.586 ± 0.017 | 1.901 ± 0.023 |
| batched | `batched_eigh` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.569 ± 0.020 | 0.138 ± 0.025 | 1.990 ± 0.672 | 5.360 ± 0.089 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.479 ± 0.026 | 0.084 ± 0.000 | 0.723 ± 0.012 | 2.100 ± 0.051 |
| batched | `batched_eigh` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.000 ± 0.111 | 0.302 ± 0.054 | 2.531 ± 0.011 | 6.666 ± 0.129 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.424 ± 0.023 | 0.010 ± 0.003 | 0.182 ± 0.026 | 1.566 ± 0.021 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.371 ± 0.035 | 0.025 ± 0.000 | 0.324 ± 0.052 | 1.930 ± 0.219 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.383 ± 0.051 | 0.010 ± 0.000 | 0.339 ± 0.065 | 2.232 ± 0.276 |
| batched | `batched_matmul_ikb_kjb_ijb` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.415 ± 0.218 | 0.026 ± 0.000 | 0.827 ± 0.003 | 2.928 ± 0.068 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.414 ± 0.030 | 0.030 ± 0.000 | 0.375 ± 0.068 | 1.309 ± 0.018 |
| batched | `batched_qr` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.480 ± 0.011 | 0.090 ± 0.000 | 1.221 ± 0.166 | 2.421 ± 0.058 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.436 ± 0.032 | 0.040 ± 0.000 | 0.486 ± 0.031 | 1.373 ± 0.020 |
| batched | `batched_qr` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.691 ± 0.144 | 0.132 ± 0.010 | 1.645 ± 0.030 | 4.405 ± 0.102 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.455 ± 0.011 | 0.041 ± 0.001 | 0.716 ± 0.032 | 1.923 ± 0.009 |
| batched | `batched_solve` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 0.538 ± 0.010 | 0.108 ± 0.009 | 2.269 ± 0.076 | 5.734 ± 0.229 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.503 ± 0.032 | 0.044 ± 0.000 | 0.829 ± 0.029 | 2.252 ± 0.021 |
| batched | `batched_solve` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.587 ± 0.016 | 0.112 ± 0.026 | 2.795 ± 0.157 | 6.778 ± 0.237 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.450 ± 0.034 | 0.067 ± 0.003 | 0.429 ± 0.086 | 1.171 ± 0.041 |
| batched | `batched_svd` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.622 ± 0.023 | 0.233 ± 0.015 | 1.209 ± 0.024 | 2.688 ± 0.050 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.565 ± 0.042 | 0.163 ± 0.004 | 0.547 ± 0.011 | 1.473 ± 0.013 |
| batched | `batched_svd` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 1.726 ± 0.451 | 0.573 ± 0.068 | 1.821 ± 0.056 | 5.159 ± 0.051 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch16 (native batch layout)` | 0.653 ± 0.091 | 0.063 ± 0.012 | 0.376 ± 0.009 | 6.245 ± 1.589 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `2x2xbatch64 (native batch layout)` | 0.750 ± 0.038 | 0.105 ± 0.003 | 0.546 ± 0.014 | 6.378 ± 1.837 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch16 (native batch layout)` | 0.664 ± 0.023 | 0.064 ± 0.008 | 0.549 ± 0.064 | 5.996 ± 1.883 |
| batched | `grad_sum_batched_matmul_backward` | f64 | 4 | `4x4xbatch64 (native batch layout)` | 0.706 ± 0.028 | 0.107 ± 0.011 | 1.129 ± 0.025 | 6.418 ± 2.670 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch16 (native batch layout),rhs=1` | 0.922 ± 0.045 | 0.221 ± 0.008 | 0.932 ± 0.147 | 6.934 ± 1.407 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `2x2xbatch64 (native batch layout),rhs=1` | 1.030 ± 0.038 | 0.381 ± 0.005 | 2.583 ± 0.063 | 7.923 ± 3.629 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch16 (native batch layout),rhs=1` | 0.896 ± 0.059 | 0.220 ± 0.007 | 1.079 ± 0.033 | 6.037 ± 2.174 |
| batched | `grad_sum_batched_solve_backward` | f64 | 4 | `4x4xbatch64 (native batch layout),rhs=1` | 0.995 ± 0.047 | 0.402 ± 0.002 | 3.065 ± 0.063 | 8.337 ± 2.966 |
| large | `eigh` | f64 | 4 | `64x64` | 1.279 ± 0.034 | 0.410 ± 0.017 | 2.365 ± 0.092 | 6.792 ± 0.058 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | - | 8.282 ± 0.160 | 8.182 ± 0.193 | 29.690 ± 0.475 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | - | 30.525 ± 0.865 | 28.267 ± 0.312 | 88.789 ± 5.303 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | - | 7.637 ± 0.146 | 6.521 ± 0.412 | 28.563 ± 1.523 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | - | 29.283 ± 0.301 | 21.345 ± 0.344 | 88.646 ± 3.305 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | - | 5.272 ± 0.285 | 4.531 ± 0.157 | 7.324 ± 1.681 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | - | 23.892 ± 0.734 | 24.787 ± 0.528 | 23.799 ± 3.082 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | - | 4.864 ± 0.192 | 2.831 ± 0.135 | 10.714 ± 1.535 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | - | 22.522 ± 0.251 | 11.949 ± 0.296 | 30.937 ± 5.709 |
| large | `grad_sum_matmul` | f64 | 4 | `64x64` | 0.484 ± 0.017 | 0.038 ± 0.012 | 3.428 ± 0.042 | 6.150 ± 0.381 |
| large | `grad_sum_matmul_backward` | f64 | 4 | `64x64` | 0.815 ± 0.115 | 0.110 ± 0.003 | 3.477 ± 0.155 | 7.982 ± 2.240 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | - | 7.036 ± 0.282 | 4.608 ± 0.148 | 13.969 ± 0.535 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | - | 30.797 ± 1.169 | 19.190 ± 0.252 | 45.298 ± 2.553 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | - | 6.922 ± 0.567 | 4.363 ± 0.144 | 14.030 ± 0.576 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | - | 28.912 ± 0.887 | 18.894 ± 0.343 | 60.881 ± 2.649 |
| large | `grad_sum_solve_backward` | f64 | 4 | `64x64,rhs=1` | 1.338 ± 0.025 | 0.229 ± 0.007 | 1.958 ± 0.022 | 7.650 ± 2.463 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | - | 1.431 ± 0.081 | 1.591 ± 0.059 | 4.110 ± 0.063 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | - | 5.439 ± 0.142 | 3.263 ± 0.152 | 10.390 ± 0.165 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | - | 0.943 ± 0.082 | 1.724 ± 0.101 | 5.608 ± 0.187 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | - | 3.246 ± 0.078 | 4.654 ± 0.188 | 11.262 ± 0.227 |
| large | `grad_sum_svd_s_backward` | f64 | 4 | `64x64` | 2.704 ± 0.173 | 1.110 ± 0.035 | 2.804 ± 0.058 | 7.359 ± 0.529 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | - | 17.299 ± 0.165 | 18.330 ± 0.410 | 52.039 ± 0.829 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | - | 70.598 ± 0.539 | 81.509 ± 4.477 | 163.776 ± 3.663 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | - | 16.870 ± 0.145 | 16.529 ± 0.290 | 50.427 ± 0.733 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | - | 69.497 ± 0.399 | 64.075 ± 0.696 | 161.626 ± 2.779 |
| large | `matmul` | f64 | 4 | `128x128` | 0.625 ± 0.026 | 0.189 ± 0.020 | 12.134 ± 0.844 | 18.217 ± 0.657 |
| large | `matmul` | f64 | 4 | `256x256` | 1.873 ± 0.028 | 0.650 ± 0.025 | 46.294 ± 0.732 | 68.003 ± 23.556 |
| large | `matmul_rect` | f64 | 4 | `256x1024 * 1024x256` | 4.546 ± 0.096 | 2.598 ± 0.155 | 183.358 ± 2.994 | 184.613 ± 2.776 |
| large | `qr` | f64 | 4 | `64x64` | 0.628 ± 0.026 | 0.147 ± 0.008 | 2.071 ± 0.064 | 5.250 ± 0.111 |
| large | `solve` | f64 | 4 | `64x64,rhs=1` | 0.959 ± 0.014 | 0.058 ± 0.003 | 1.800 ± 0.041 | 5.540 ± 0.099 |
| large | `solve` | f64 | 4 | `64x64,rhs=16` | 0.956 ± 0.035 | 0.083 ± 0.003 | 2.425 ± 0.047 | 4.733 ± 0.073 |
| large | `solve` | f64 | 4 | `64x64,rhs=64` | 0.986 ± 0.019 | 0.142 ± 0.005 | 3.796 ± 0.046 | 6.679 ± 0.352 |
| large | `svd` | f64 | 4 | `64x64` | 1.440 ± 0.060 | 1.002 ± 0.026 | 2.784 ± 0.058 | 6.843 ± 1.586 |
| small | `eigh` | f64 | 4 | `2x2` | 0.403 ± 0.021 | 0.009 ± 0.001 | 0.121 ± 0.033 | 0.390 ± 0.018 |
| small | `eigh` | f64 | 4 | `4x4` | 0.409 ± 0.047 | 0.014 ± 0.000 | 0.123 ± 0.003 | 0.717 ± 0.034 |
| small | `eigh` | f64 | 4 | `8x8` | 0.416 ± 0.021 | 0.020 ± 0.001 | 0.153 ± 0.001 | 0.801 ± 0.017 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `2x2` | 0.413 ± 0.018 | 0.005 ± 0.001 | 0.145 ± 0.026 | 0.560 ± 0.026 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `4x4` | 0.381 ± 0.015 | 0.005 ± 0.001 | 0.126 ± 0.005 | 1.417 ± 0.015 |
| small | `einsum_ij_jk_ik` | f64 | 4 | `8x8` | 0.369 ± 0.031 | 0.005 ± 0.005 | 0.171 ± 0.010 | 1.812 ± 0.006 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | - | 0.105 ± 0.021 | 0.338 ± 0.055 | 1.549 ± 0.023 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | - | 0.125 ± 0.020 | 0.359 ± 0.020 | 1.591 ± 0.125 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | - | 0.144 ± 0.008 | 0.378 ± 0.012 | 1.801 ± 0.049 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | - | 0.124 ± 0.018 | 0.325 ± 0.014 | 3.505 ± 0.097 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | - | 0.132 ± 0.018 | 0.331 ± 0.012 | 3.566 ± 0.358 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | - | 0.144 ± 0.030 | 0.343 ± 0.019 | 4.753 ± 0.162 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | - | 0.113 ± 0.016 | 0.523 ± 0.013 | 2.024 ± 0.055 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | - | 0.111 ± 0.012 | 0.520 ± 0.014 | 1.829 ± 0.020 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | - | 0.131 ± 0.023 | 0.541 ± 0.118 | 3.108 ± 0.051 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | - | 0.126 ± 0.015 | 0.495 ± 0.049 | 5.214 ± 1.037 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | - | 0.121 ± 0.007 | 0.485 ± 0.039 | 5.200 ± 0.996 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | - | 0.128 ± 0.011 | 0.471 ± 0.060 | 5.607 ± 1.994 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `2x2` | 0.697 ± 0.047 | 0.053 ± 0.008 | 0.232 ± 0.017 | 5.252 ± 0.657 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `4x4` | 0.640 ± 0.009 | 0.048 ± 0.016 | 0.256 ± 0.034 | 5.305 ± 1.194 |
| small | `grad_sum_matmul_backward` | f64 | 4 | `8x8` | 0.639 ± 0.036 | 0.054 ± 0.021 | 0.281 ± 0.008 | 5.289 ± 0.976 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | - | 0.296 ± 0.032 | 0.426 ± 0.030 | 3.184 ± 0.026 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | - | 0.293 ± 0.014 | 0.416 ± 0.026 | 2.714 ± 0.083 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | - | 0.292 ± 0.039 | 0.408 ± 0.020 | 2.686 ± 0.073 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | - | 0.582 ± 0.025 | 0.467 ± 0.048 | 6.398 ± 1.807 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | - | 0.579 ± 0.014 | 0.432 ± 0.019 | 6.596 ± 1.134 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | - | 0.579 ± 0.034 | 0.440 ± 0.012 | 6.953 ± 1.089 |
| small | `grad_sum_solve_backward` | f64 | 4 | `2x2,rhs=1` | 0.847 ± 0.087 | 0.141 ± 0.033 | 0.336 ± 0.048 | 5.873 ± 0.639 |
| small | `grad_sum_solve_backward` | f64 | 4 | `4x4,rhs=1` | 0.837 ± 0.083 | 0.145 ± 0.016 | 0.380 ± 0.113 | 4.363 ± 1.030 |
| small | `grad_sum_solve_backward` | f64 | 4 | `8x8,rhs=1` | 0.855 ± 0.079 | 0.141 ± 0.012 | 0.367 ± 0.011 | 5.442 ± 1.327 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | - | 0.078 ± 0.005 | 0.696 ± 0.019 | 2.348 ± 0.068 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | - | 0.076 ± 0.006 | 0.697 ± 0.058 | 2.269 ± 0.071 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | - | 0.071 ± 0.007 | 0.758 ± 0.075 | 2.368 ± 0.023 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | - | 0.103 ± 0.012 | 0.451 ± 0.014 | 4.571 ± 1.193 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | - | 0.100 ± 0.012 | 0.453 ± 0.011 | 5.176 ± 0.575 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | - | 0.103 ± 0.003 | 0.491 ± 0.017 | 5.206 ± 0.289 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `2x2` | 1.236 ± 0.051 | 0.087 ± 0.018 | 0.236 ± 0.031 | 5.525 ± 0.673 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `4x4` | 1.279 ± 0.171 | 0.098 ± 0.008 | 0.265 ± 0.024 | 5.624 ± 0.793 |
| small | `grad_sum_svd_s_backward` | f64 | 4 | `8x8` | 1.278 ± 0.042 | 0.118 ± 0.016 | 0.303 ± 0.030 | 5.957 ± 0.882 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | - | 0.069 ± 0.003 | 0.438 ± 0.010 | 1.871 ± 0.064 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | - | 0.085 ± 0.008 | 0.490 ± 0.052 | 1.884 ± 0.061 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | - | 0.097 ± 0.011 | 0.466 ± 0.014 | 1.909 ± 0.055 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | - | 0.087 ± 0.018 | 0.360 ± 0.061 | 4.036 ± 0.937 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | - | 0.092 ± 0.006 | 0.350 ± 0.013 | 5.083 ± 0.392 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | - | 0.109 ± 0.013 | 0.374 ± 0.013 | 4.634 ± 0.959 |
| small | `matmul` | f64 | 4 | `2x2` | 0.464 ± 0.091 | 0.005 ± 0.001 | 0.096 ± 0.024 | 0.349 ± 0.036 |
| small | `matmul` | f64 | 4 | `4x4` | 0.393 ± 0.061 | 0.005 ± 0.003 | 0.092 ± 0.001 | 0.758 ± 0.012 |
| small | `matmul` | f64 | 4 | `8x8` | 0.369 ± 0.036 | 0.005 ± 0.001 | 0.131 ± 0.003 | 1.181 ± 0.020 |
| small | `qr` | f64 | 4 | `2x2` | 0.436 ± 0.026 | 0.008 ± 0.000 | 0.089 ± 0.015 | 0.255 ± 0.004 |
| small | `qr` | f64 | 4 | `4x4` | 0.399 ± 0.038 | 0.009 ± 0.001 | 0.095 ± 0.020 | 0.568 ± 0.016 |
| small | `qr` | f64 | 4 | `8x8` | 0.385 ± 0.017 | 0.011 ± 0.001 | 0.116 ± 0.007 | 0.674 ± 0.022 |
| small | `solve` | f64 | 4 | `2x2,rhs=1` | 0.482 ± 0.136 | 0.018 ± 0.001 | 0.168 ± 0.013 | 0.630 ± 0.017 |
| small | `solve` | f64 | 4 | `2x2,rhs=4` | 0.499 ± 0.042 | 0.020 ± 0.001 | 0.163 ± 0.021 | 0.651 ± 0.028 |
| small | `solve` | f64 | 4 | `4x4,rhs=1` | 0.436 ± 0.029 | 0.019 ± 0.001 | 0.182 ± 0.024 | 0.838 ± 0.020 |
| small | `solve` | f64 | 4 | `4x4,rhs=4` | 0.439 ± 0.009 | 0.021 ± 0.004 | 0.175 ± 0.012 | 0.844 ± 0.013 |
| small | `solve` | f64 | 4 | `8x8,rhs=1` | 0.464 ± 0.043 | 0.019 ± 0.001 | 0.207 ± 0.021 | 0.938 ± 0.011 |
| small | `solve` | f64 | 4 | `8x8,rhs=4` | 0.438 ± 0.052 | 0.021 ± 0.000 | 0.202 ± 0.003 | 0.872 ± 0.007 |
| small | `svd` | f64 | 4 | `2x2` | 0.418 ± 0.030 | 0.013 ± 0.001 | 0.103 ± 0.002 | 0.280 ± 0.031 |
| small | `svd` | f64 | 4 | `4x4` | 0.411 ± 0.019 | 0.019 ± 0.001 | 0.111 ± 0.003 | 0.543 ± 0.018 |
| small | `svd` | f64 | 4 | `8x8` | 0.421 ± 0.021 | 0.034 ± 0.006 | 0.144 ± 0.004 | 0.679 ± 0.021 |

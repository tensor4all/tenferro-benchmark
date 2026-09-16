# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260913/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-16T02:18:36.821467+00:00`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92000`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Small cases (n=2, 4, 8) are single-call diagnostics, not batched shared-session small-work comparisons. Public API session entry remains included; these rows must not be interpreted as isolated kernel or shared-session overhead. This suite checks successful execution only; AD outputs are not numerically compared.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.426 ± 0.052 | 2.999 ± 0.008 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 7.002 ± 0.037 | 6.390 ± 0.008 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.417 ± 0.097 | 3.033 ± 0.007 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.985 ± 0.166 | 6.405 ± 0.016 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.857 ± 0.014 | 1.146 ± 0.001 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.647 ± 0.043 | 2.426 ± 0.007 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.991 ± 0.025 | 0.903 ± 0.003 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.827 ± 0.053 | 1.927 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.782 ± 0.017 | 1.491 ± 0.007 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.207 ± 0.005 | 3.254 ± 0.014 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.650 ± 0.048 | 1.480 ± 0.009 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.359 ± 0.096 | 3.230 ± 0.025 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 2.194 ± 0.021 | 0.944 ± 0.034 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.927 ± 0.086 | 1.828 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.483 ± 0.215 | 1.647 ± 0.067 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.975 ± 0.010 | 3.317 ± 0.079 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.202 ± 0.009 | 12.742 ± 0.022 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.923 ± 0.151 | 33.603 ± 0.008 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.001 ± 0.199 | 12.827 ± 0.030 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.820 ± 0.134 | 33.523 ± 0.065 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.711 ± 0.008 | 0.399 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.718 ± 0.016 | 0.401 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.788 ± 0.078 | 0.440 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.669 ± 0.009 | 0.445 ± 0.012 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.799 ± 0.005 | 0.457 ± 0.020 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.651 ± 0.064 | 0.505 ± 0.038 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 2.086 ± 0.017 | 0.594 ± 0.006 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.469 ± 0.008 | 0.591 ± 0.008 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.090 ± 0.007 | 0.601 ± 0.011 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.193 ± 0.015 | 0.864 ± 0.167 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 1.156 ± 0.001 | 0.765 ± 0.020 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.029 ± 0.003 | 0.771 ± 0.019 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.671 ± 0.051 | 0.448 ± 0.020 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.380 ± 0.008 | 0.449 ± 0.009 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.630 ± 0.018 | 0.535 ± 0.011 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.373 ± 0.019 | 0.585 ± 0.021 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.525 ± 0.121 | 0.752 ± 0.019 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.481 ± 0.010 | 0.806 ± 0.028 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 2.628 ± 0.057 | 0.584 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.802 ± 0.139 | 0.581 ± 0.103 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.583 ± 0.007 | 0.582 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.712 ± 0.014 | 0.721 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.761 ± 0.006 | 0.733 ± 0.070 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.912 ± 0.086 | 0.720 ± 0.049 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.740 ± 0.011 | 0.540 ± 0.013 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.030 ± 0.004 | 0.582 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.072 ± 0.010 | 0.627 ± 0.012 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.714 ± 0.005 | 0.431 ± 0.070 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.951 ± 0.018 | 0.527 ± 0.018 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.076 ± 0.005 | 0.554 ± 0.018 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

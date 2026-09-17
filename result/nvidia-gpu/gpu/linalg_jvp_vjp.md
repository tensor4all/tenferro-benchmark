# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260913/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-17T21:09:06.847171+00:00`
- tenferro-rs commit: `292cdffef8d910abb88dca44cd3c928bc6051005`

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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.519 ± 0.124 | 3.083 ± 0.025 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.777 ± 0.108 | 6.390 ± 0.057 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.503 ± 0.078 | 3.218 ± 0.033 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.732 ± 0.167 | 6.599 ± 0.144 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.967 ± 0.011 | 1.168 ± 0.004 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.972 ± 0.017 | 2.446 ± 0.005 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.884 ± 0.004 | 1.029 ± 0.050 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.642 ± 0.371 | 1.953 ± 0.012 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.517 ± 0.020 | 1.512 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.588 ± 0.054 | 3.270 ± 0.010 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.545 ± 0.174 | 1.473 ± 0.009 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.396 ± 0.055 | 3.240 ± 0.019 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 2.393 ± 0.198 | 0.980 ± 0.007 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.683 ± 0.056 | 1.861 ± 0.008 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.490 ± 0.129 | 1.868 ± 0.106 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.848 ± 0.027 | 3.528 ± 0.114 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.199 ± 0.060 | 12.593 ± 0.026 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 33.165 ± 0.209 | 33.604 ± 0.077 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.363 ± 0.043 | 12.607 ± 0.053 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.996 ± 0.052 | 33.630 ± 0.107 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.722 ± 0.011 | 0.400 ± 0.007 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.733 ± 0.015 | 0.465 ± 0.014 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.658 ± 0.052 | 0.510 ± 0.007 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.718 ± 0.015 | 0.567 ± 0.014 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.603 ± 0.006 | 0.483 ± 0.061 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.635 ± 0.006 | 0.535 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.045 ± 0.161 | 0.706 ± 0.056 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.319 ± 0.017 | 0.709 ± 0.007 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.453 ± 0.012 | 0.710 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.222 ± 0.007 | 0.963 ± 0.022 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 1.252 ± 0.009 | 0.816 ± 0.088 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.262 ± 0.009 | 0.827 ± 0.153 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 1.261 ± 0.011 | 0.539 ± 0.025 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.480 ± 0.011 | 0.528 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.491 ± 0.010 | 0.534 ± 0.017 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.442 ± 0.079 | 0.723 ± 0.084 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.439 ± 0.013 | 0.639 ± 0.145 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.600 ± 0.011 | 0.820 ± 0.025 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 2.657 ± 0.022 | 0.691 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 2.039 ± 0.125 | 0.686 ± 0.022 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.768 ± 0.008 | 0.686 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.828 ± 0.357 | 0.632 ± 0.019 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.477 ± 0.015 | 0.788 ± 0.026 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 2.064 ± 0.017 | 0.773 ± 0.053 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.731 ± 0.010 | 0.530 ± 0.009 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.918 ± 0.012 | 0.669 ± 0.020 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.011 ± 0.011 | 0.721 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.635 ± 0.009 | 0.416 ± 0.059 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.887 ± 0.025 | 0.508 ± 0.022 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.989 ± 0.015 | 0.624 ± 0.036 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

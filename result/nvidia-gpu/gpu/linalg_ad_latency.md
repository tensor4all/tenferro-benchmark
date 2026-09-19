# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260919/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-09-19T15:15:14.205198+00:00`
- tenferro-rs commit: `d7a8c60caedac9aa3e630eea793d606df02dfc09`

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
Numerically verified rows record an untimed backend-primal central-difference reference (h=1e-5), comparing the JVP or the VJP dotted with the deterministic tangent. This checks one direction, not every gradient component. Legacy rows without a reference_backend are execution-only, even if their status says passed.
Small cases (n=2, 4, 8) are single-call diagnostics, not batched shared-session small-work comparisons. Public API session entry remains included; these rows must not be interpreted as isolated kernel or shared-session overhead.
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.471 ± 0.010 | 0.402 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.560 ± 0.031 | 0.412 ± 0.019 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.602 ± 0.090 | 0.448 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.558 ± 0.015 | 0.556 ± 0.021 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.492 ± 0.008 | 0.448 ± 0.013 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.532 ± 0.009 | 0.476 ± 0.068 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.941 ± 0.044 | 0.607 ± 0.115 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.886 ± 0.011 | 0.615 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.796 ± 0.009 | 0.620 ± 0.113 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.786 ± 0.007 | 0.951 ± 0.062 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.793 ± 0.011 | 0.770 ± 0.020 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.805 ± 0.003 | 0.776 ± 0.027 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.169 ± 0.046 | 0.460 ± 0.008 |
| small | [`grad_sum_qr_jvp`](../../../notes/nvidia-gpu/gpu/linalg_ad_latency.md#linalg_ad_small_grad_sum_qr_jvp_f64_4) | f64 | `4x4` | 0.938 ± 0.007 | 0.456 ± 0.010 |
| small | [`grad_sum_qr_jvp`](../../../notes/nvidia-gpu/gpu/linalg_ad_latency.md#linalg_ad_small_grad_sum_qr_jvp_f64_8) | f64 | `8x8` | 0.969 ± 0.010 | 0.461 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.125 ± 0.429 | 0.740 ± 0.165 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 0.934 ± 0.012 | 0.885 ± 0.197 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 0.955 ± 0.008 | 0.763 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.375 ± 0.017 | 0.585 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.137 ± 0.011 | 0.591 ± 0.014 |
| small | [`grad_sum_solve_jvp`](../../../notes/nvidia-gpu/gpu/linalg_ad_latency.md#linalg_ad_small_grad_sum_solve_jvp_f64_8) | f64 | `8x8,rhs=1` | 1.345 ± 0.014 | 0.585 ± 0.017 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.317 ± 0.052 | 0.713 ± 0.028 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.142 ± 0.008 | 0.726 ± 0.041 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.201 ± 0.180 | 0.722 ± 0.028 |
| small | [`grad_sum_svd_s_jvp`](../../../notes/nvidia-gpu/gpu/linalg_ad_latency.md#linalg_ad_small_grad_sum_svd_s_jvp_f64_2) | f64 | `2x2` | 1.012 ± 0.065 | 0.612 ± 0.100 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.701 ± 0.050 | 0.594 ± 0.015 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.795 ± 0.041 | 0.631 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.623 ± 0.012 | 0.435 ± 0.060 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.748 ± 0.013 | 0.555 ± 0.148 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.859 ± 0.101 | 0.614 ± 0.069 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/workspaces/t4a-2010-bench-gpu/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-10-06T19:48:20.706119+00:00`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

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
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.649 ± 0.013 | 0.536 ± 0.032 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.673 ± 0.017 | 0.507 ± 0.028 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.701 ± 0.013 | 0.546 ± 0.038 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.641 ± 0.017 | 0.608 ± 0.109 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.661 ± 0.013 | 0.551 ± 0.035 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.699 ± 0.009 | 0.591 ± 0.118 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.549 ± 0.047 | 0.819 ± 0.059 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.962 ± 0.006 | 0.782 ± 0.039 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.974 ± 0.005 | 0.780 ± 0.019 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.575 ± 0.160 | 0.978 ± 0.102 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.969 ± 0.011 | 0.949 ± 0.031 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.044 ± 0.009 | 0.920 ± 0.038 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.020 ± 0.052 | 0.622 ± 0.019 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.222 ± 0.015 | 0.612 ± 0.043 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.242 ± 0.014 | 0.589 ± 0.021 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.884 ± 0.028 | 0.960 ± 0.294 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.224 ± 0.008 | 0.949 ± 0.024 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.234 ± 0.019 | 0.923 ± 0.024 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 2.250 ± 0.037 | 0.739 ± 0.031 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.414 ± 0.007 | 0.736 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.427 ± 0.009 | 0.749 ± 0.041 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 2.329 ± 0.114 | 0.873 ± 0.056 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.431 ± 0.009 | 0.878 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.440 ± 0.010 | 0.901 ± 0.016 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.621 ± 0.014 | 0.621 ± 0.014 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.070 ± 0.021 | 0.741 ± 0.024 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.907 ± 0.047 | 0.786 ± 0.027 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 1.031 ± 0.010 | 0.519 ± 0.041 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.832 ± 0.015 | 0.670 ± 0.120 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.916 ± 0.022 | 0.673 ± 0.018 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

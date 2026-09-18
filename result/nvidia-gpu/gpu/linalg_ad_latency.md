# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260913/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-09-18T00:09:39.896463+00:00`
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
This suite exists to measure single-call latency and per-op overhead: the device kernels are a small fraction of each row, so the numbers are not GPU throughput. GPU-sized AD results live in `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.693 ± 0.009 | 0.406 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.752 ± 0.069 | 0.478 ± 0.013 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.862 ± 0.019 | 0.451 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.686 ± 0.011 | 0.558 ± 0.098 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.806 ± 0.016 | 0.505 ± 0.017 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.856 ± 0.096 | 0.503 ± 0.047 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.064 ± 0.008 | 0.718 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.454 ± 0.029 | 0.719 ± 0.014 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.525 ± 0.140 | 0.718 ± 0.008 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.181 ± 0.004 | 0.737 ± 0.092 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 1.362 ± 0.117 | 0.846 ± 0.108 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.283 ± 0.019 | 0.826 ± 0.022 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 1.367 ± 0.016 | 0.551 ± 0.003 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.644 ± 0.163 | 0.534 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.516 ± 0.152 | 0.462 ± 0.012 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.367 ± 0.039 | 0.881 ± 0.153 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.377 ± 0.012 | 0.814 ± 0.032 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.457 ± 0.012 | 0.757 ± 0.021 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.763 ± 0.017 | 0.704 ± 0.015 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.767 ± 0.004 | 0.702 ± 0.031 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.961 ± 0.194 | 0.701 ± 0.020 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.818 ± 0.105 | 0.791 ± 0.021 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 2.015 ± 0.021 | 0.731 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 2.034 ± 0.029 | 0.633 ± 0.031 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 1.230 ± 0.022 | 0.541 ± 0.015 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 1.102 ± 0.016 | 0.678 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 1.066 ± 0.015 | 0.637 ± 0.005 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.710 ± 0.081 | 0.444 ± 0.059 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.951 ± 0.017 | 0.570 ± 0.016 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.039 ± 0.014 | 0.574 ± 0.015 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

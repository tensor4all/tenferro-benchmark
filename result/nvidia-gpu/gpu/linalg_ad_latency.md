# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_latency`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260919/benchmarks/gpu/linalg_ad_latency.yaml`
- Timestamp: `2026-09-19T08:26:05.008726+00:00`
- tenferro-rs commit: `40e24634f9aa814db82efd9faae5e99a8fcee8c4`

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
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.608 ± 0.007 | 0.404 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.745 ± 0.009 | 0.407 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.867 ± 0.080 | 0.450 ± 0.009 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.721 ± 0.005 | 0.564 ± 0.018 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.752 ± 0.008 | 0.446 ± 0.002 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.787 ± 0.068 | 0.489 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 1.023 ± 0.004 | 0.610 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 1.327 ± 0.078 | 0.605 ± 0.012 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.302 ± 0.024 | 0.615 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 1.023 ± 0.006 | 0.894 ± 0.072 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 1.291 ± 0.119 | 0.771 ± 0.039 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 1.293 ± 0.017 | 0.779 ± 0.023 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 2.531 ± 1.289 | 0.462 ± 0.009 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.393 ± 0.201 | 0.530 ± 0.089 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.224 ± 0.013 | 0.472 ± 0.096 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.446 ± 0.014 | 0.762 ± 0.189 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.611 ± 0.072 | 0.759 ± 0.023 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.509 ± 0.107 | 0.762 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.539 ± 0.079 | 0.594 ± 0.115 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.526 ± 0.019 | 0.590 ± 0.110 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.556 ± 0.012 | 0.585 ± 0.009 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.934 ± 0.301 | 0.792 ± 0.020 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.982 ± 0.107 | 0.789 ± 0.037 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.923 ± 0.150 | 0.728 ± 0.019 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.697 ± 0.004 | 0.543 ± 0.059 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.983 ± 0.039 | 0.589 ± 0.008 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.937 ± 0.116 | 0.730 ± 0.018 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.641 ± 0.003 | 0.450 ± 0.040 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.899 ± 0.017 | 0.689 ± 0.044 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 1.019 ± 0.051 | 0.633 ± 0.073 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

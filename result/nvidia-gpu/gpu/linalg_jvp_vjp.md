# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260919/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-19T08:25:11.356784+00:00`
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

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.441 ± 0.140 | 3.080 ± 0.048 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.793 ± 0.083 | 6.385 ± 0.035 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.464 ± 0.112 | 3.066 ± 0.062 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.787 ± 0.163 | 6.394 ± 0.044 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.421 ± 0.030 | 1.143 ± 0.002 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.777 ± 0.124 | 2.418 ± 0.016 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 2.009 ± 0.018 | 0.903 ± 0.005 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.663 ± 0.184 | 1.927 ± 0.002 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 3.947 ± 0.430 | 1.488 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.004 ± 0.360 | 3.270 ± 0.022 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.510 ± 0.010 | 1.474 ± 0.034 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.569 ± 0.113 | 3.234 ± 0.022 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 3.233 ± 0.018 | 0.942 ± 0.031 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.833 ± 0.187 | 1.870 ± 0.045 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.339 ± 0.188 | 1.580 ± 0.009 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.947 ± 0.035 | 3.320 ± 0.082 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.201 ± 0.024 | 12.477 ± 0.100 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.804 ± 0.085 | 33.444 ± 0.085 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.003 ± 0.134 | 12.370 ± 0.087 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.988 ± 0.059 | 33.319 ± 0.022 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

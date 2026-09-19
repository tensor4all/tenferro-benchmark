# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260919/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-19T15:16:20.323362+00:00`
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

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.224 ± 0.038 | 3.037 ± 0.007 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.609 ± 0.027 | 6.428 ± 0.063 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.474 ± 0.115 | 3.228 ± 0.036 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.897 ± 0.096 | 6.390 ± 0.008 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.728 ± 0.167 | 1.143 ± 0.002 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.571 ± 0.041 | 2.416 ± 0.008 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.808 ± 0.119 | 0.910 ± 0.069 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.692 ± 0.096 | 1.920 ± 0.004 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.140 ± 0.261 | 1.494 ± 0.010 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 3.729 ± 0.054 | 3.248 ± 0.006 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.289 ± 0.032 | 1.463 ± 0.016 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 3.851 ± 0.070 | 3.216 ± 0.008 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 2.819 ± 0.025 | 0.936 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.024 ± 0.010 | 1.827 ± 0.016 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.282 ± 0.169 | 1.585 ± 0.110 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.038 ± 0.017 | 3.354 ± 0.083 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 11.928 ± 0.048 | 12.486 ± 0.032 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.780 ± 0.239 | 33.491 ± 0.079 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.069 ± 0.012 | 12.501 ± 0.042 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.656 ± 0.016 | 33.344 ± 0.037 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

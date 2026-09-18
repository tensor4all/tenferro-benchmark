# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260913/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-18T00:07:14.421561+00:00`
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

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.559 ± 0.058 | 3.081 ± 0.045 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.796 ± 0.082 | 6.414 ± 0.083 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.522 ± 0.025 | 3.252 ± 0.100 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.793 ± 0.174 | 6.544 ± 0.025 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.936 ± 0.013 | 1.165 ± 0.003 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.765 ± 0.018 | 2.440 ± 0.024 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.856 ± 0.096 | 1.041 ± 0.033 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.782 ± 0.076 | 1.950 ± 0.019 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 3.569 ± 0.568 | 1.511 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.417 ± 0.082 | 3.264 ± 0.012 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.509 ± 0.041 | 1.466 ± 0.024 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.621 ± 0.096 | 3.250 ± 0.030 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 2.541 ± 0.421 | 0.975 ± 0.003 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 3.542 ± 0.464 | 1.873 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.104 ± 0.008 | 1.762 ± 0.064 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.100 ± 0.019 | 3.533 ± 0.141 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.183 ± 0.055 | 12.516 ± 0.024 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.945 ± 0.094 | 33.529 ± 0.079 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.203 ± 0.016 | 12.542 ± 0.051 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.995 ± 0.074 | 33.613 ± 0.071 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

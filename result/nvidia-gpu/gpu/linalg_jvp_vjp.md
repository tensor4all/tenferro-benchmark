# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/t4a-2010-bench-gpu/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-10-06T19:47:20.990197+00:00`
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

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | PyTorch CUDA |
|---|---|---:|---|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.287 ± 0.033 | 3.116 ± 0.020 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.749 ± 0.054 | 6.565 ± 0.020 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.292 ± 0.021 | 3.269 ± 0.034 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.771 ± 0.032 | 6.538 ± 0.059 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.157 ± 0.038 | 1.163 ± 0.003 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.495 ± 0.024 | 2.450 ± 0.007 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.514 ± 0.025 | 0.926 ± 0.003 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.603 ± 0.027 | 1.950 ± 0.005 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 3.138 ± 0.035 | 1.502 ± 0.005 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.291 ± 0.046 | 3.292 ± 0.022 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 3.188 ± 0.091 | 1.468 ± 0.010 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.129 ± 0.270 | 3.252 ± 0.010 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 2.688 ± 0.023 | 0.984 ± 0.012 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.891 ± 0.025 | 1.863 ± 0.015 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.692 ± 0.019 | 1.714 ± 0.010 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.989 ± 0.020 | 3.298 ± 0.036 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.080 ± 0.039 | 12.639 ± 0.033 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.948 ± 0.142 | 33.675 ± 0.087 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.122 ± 0.095 | 12.705 ± 0.050 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 33.070 ± 0.028 | 33.702 ± 0.105 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

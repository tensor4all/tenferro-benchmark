# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspace/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-18T15:05:34.258416+00:00`
- tenferro-rs commit: `cf971d0f18d0357133da97dae66802e26bd56e2f`

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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.423 ± 0.100 | 3.037 ± 0.019 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.750 ± 0.173 | 6.436 ± 0.009 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.539 ± 0.109 | 3.207 ± 0.025 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.775 ± 0.101 | 6.480 ± 0.113 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 2.802 ± 0.022 | 1.142 ± 0.002 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.599 ± 0.010 | 2.435 ± 0.007 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.999 ± 0.015 | 0.925 ± 0.087 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.761 ± 0.007 | 1.927 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 3.962 ± 0.024 | 1.484 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.117 ± 0.163 | 3.267 ± 0.014 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.496 ± 0.015 | 1.463 ± 0.004 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.413 ± 0.047 | 3.225 ± 0.013 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 3.374 ± 0.082 | 0.939 ± 0.035 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.934 ± 0.017 | 1.867 ± 0.005 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 2.316 ± 0.014 | 1.714 ± 0.014 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 3.082 ± 0.241 | 3.388 ± 0.136 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.181 ± 0.165 | 12.479 ± 0.067 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.988 ± 0.024 | 33.414 ± 0.021 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.205 ± 0.008 | 12.497 ± 0.043 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 33.089 ± 0.147 | 33.476 ± 0.047 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

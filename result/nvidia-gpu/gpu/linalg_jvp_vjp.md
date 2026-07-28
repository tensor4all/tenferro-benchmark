# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/workspaces/tenferro-benchmark/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-07-28T03:54:19.679110+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.400 ± 0.061 | 3.026 ± 0.005 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.558 ± 0.095 | 6.509 ± 0.015 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.389 ± 0.036 | 3.192 ± 0.016 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.584 ± 0.057 | 6.543 ± 0.123 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.701 ± 0.097 | 1.146 ± 0.006 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.374 ± 0.032 | 2.430 ± 0.008 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.643 ± 0.026 | 0.984 ± 0.008 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.430 ± 0.032 | 1.927 ± 0.006 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.395 ± 0.227 | 1.494 ± 0.009 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.149 ± 0.114 | 3.243 ± 0.003 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.383 ± 0.073 | 1.457 ± 0.013 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 4.180 ± 0.046 | 3.234 ± 0.016 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.555 ± 0.010 | 0.938 ± 0.005 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.654 ± 0.025 | 1.857 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.541 ± 0.008 | 1.806 ± 0.116 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.701 ± 0.025 | 3.376 ± 0.056 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.135 ± 0.105 | 12.709 ± 0.079 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.790 ± 0.162 | 33.516 ± 0.075 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 11.997 ± 0.005 | 12.766 ± 0.018 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.811 ± 0.065 | 33.587 ± 0.086 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.621 ± 0.011 | 0.403 ± 0.010 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.630 ± 0.012 | 0.405 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.541 ± 0.005 | 0.440 ± 0.005 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.603 ± 0.038 | 0.554 ± 0.027 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.686 ± 0.036 | 0.390 ± 0.120 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.556 ± 0.020 | 0.494 ± 0.019 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.864 ± 0.020 | 0.604 ± 0.017 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.886 ± 0.008 | 0.607 ± 0.106 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 0.701 ± 0.005 | 0.604 ± 0.008 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.650 ± 0.007 | 0.904 ± 0.029 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.870 ± 0.008 | 0.547 ± 0.009 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.763 ± 0.097 | 0.769 ± 0.017 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 1.008 ± 0.113 | 0.447 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 1.195 ± 0.222 | 0.450 ± 0.006 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 1.243 ± 0.026 | 0.448 ± 0.009 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 1.078 ± 0.044 | 0.643 ± 0.299 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 1.082 ± 0.087 | 0.742 ± 0.025 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 1.098 ± 0.048 | 0.757 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.344 ± 0.112 | 0.573 ± 0.023 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.349 ± 0.015 | 0.570 ± 0.010 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.119 ± 0.135 | 0.577 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.281 ± 0.022 | 0.735 ± 0.080 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.309 ± 0.014 | 0.707 ± 0.013 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.142 ± 0.116 | 0.720 ± 0.052 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.539 ± 0.008 | 0.530 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.766 ± 0.010 | 0.585 ± 0.006 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.860 ± 0.017 | 0.637 ± 0.128 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.640 ± 0.001 | 0.445 ± 0.048 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.768 ± 0.018 | 0.522 ± 0.016 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.870 ± 0.017 | 0.611 ± 0.055 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

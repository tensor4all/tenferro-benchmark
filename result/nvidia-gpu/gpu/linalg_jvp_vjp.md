# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_jvp_vjp`
- Suite file: `/home/shinaoka/tensor4all/.worktrees/bench-gpu-20260913/benchmarks/gpu/linalg_jvp_vjp.yaml`
- Timestamp: `2026-09-13T08:53:23.540253+00:00`
- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

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
| large | `grad_sum_eigh_jvp` | f64 | `256x256` | 3.356 ± 0.093 | 2.993 ± 0.013 |
| large | `grad_sum_eigh_jvp` | f64 | `512x512` | 6.791 ± 0.065 | 6.434 ± 0.025 |
| large | `grad_sum_eigh_vjp` | f64 | `256x256` | 3.346 ± 0.056 | 3.106 ± 0.028 |
| large | `grad_sum_eigh_vjp` | f64 | `512x512` | 6.792 ± 0.204 | 6.423 ± 0.119 |
| large | `grad_sum_lu_jvp` | f64 | `256x256` | 1.787 ± 0.019 | 1.157 ± 0.005 |
| large | `grad_sum_lu_jvp` | f64 | `512x512` | 2.474 ± 0.153 | 2.436 ± 0.010 |
| large | `grad_sum_lu_vjp` | f64 | `256x256` | 1.719 ± 0.013 | 0.920 ± 0.003 |
| large | `grad_sum_lu_vjp` | f64 | `512x512` | 2.528 ± 0.038 | 1.927 ± 0.010 |
| large | `grad_sum_qr_jvp` | f64 | `256x256` | 2.451 ± 0.020 | 1.505 ± 0.010 |
| large | `grad_sum_qr_jvp` | f64 | `512x512` | 4.093 ± 0.091 | 3.257 ± 0.028 |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | 2.449 ± 0.039 | 1.464 ± 0.011 |
| large | `grad_sum_qr_vjp` | f64 | `512x512` | 3.990 ± 0.006 | 3.230 ± 0.038 |
| large | `grad_sum_solve_jvp` | f64 | `256x256,rhs=1` | 1.707 ± 0.020 | 0.979 ± 0.009 |
| large | `grad_sum_solve_jvp` | f64 | `512x512,rhs=1` | 2.700 ± 0.024 | 1.863 ± 0.011 |
| large | `grad_sum_solve_vjp` | f64 | `256x256,rhs=1` | 1.903 ± 0.021 | 1.841 ± 0.050 |
| large | `grad_sum_solve_vjp` | f64 | `512x512,rhs=1` | 2.615 ± 0.050 | 3.315 ± 0.017 |
| large | `grad_sum_svd_s_jvp` | f64 | `256x256` | 12.173 ± 0.213 | 12.796 ± 0.012 |
| large | `grad_sum_svd_s_jvp` | f64 | `512x512` | 32.796 ± 0.099 | 33.487 ± 0.011 |
| large | `grad_sum_svd_s_vjp` | f64 | `256x256` | 12.001 ± 0.068 | 12.783 ± 0.033 |
| large | `grad_sum_svd_s_vjp` | f64 | `512x512` | 32.794 ± 0.131 | 33.571 ± 0.189 |
| small | `grad_sum_eigh_jvp` | f64 | `2x2` | 0.559 ± 0.017 | 0.399 ± 0.008 |
| small | `grad_sum_eigh_jvp` | f64 | `4x4` | 0.490 ± 0.010 | 0.401 ± 0.011 |
| small | `grad_sum_eigh_jvp` | f64 | `8x8` | 0.534 ± 0.002 | 0.443 ± 0.059 |
| small | `grad_sum_eigh_vjp` | f64 | `2x2` | 0.461 ± 0.011 | 0.553 ± 0.116 |
| small | `grad_sum_eigh_vjp` | f64 | `4x4` | 0.528 ± 0.091 | 0.450 ± 0.011 |
| small | `grad_sum_eigh_vjp` | f64 | `8x8` | 0.522 ± 0.075 | 0.482 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | `2x2` | 0.927 ± 0.093 | 0.596 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | `4x4` | 0.835 ± 0.003 | 0.608 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | `8x8` | 1.000 ± 0.066 | 0.601 ± 0.105 |
| small | `grad_sum_lu_vjp` | f64 | `2x2` | 0.752 ± 0.008 | 0.784 ± 0.148 |
| small | `grad_sum_lu_vjp` | f64 | `4x4` | 0.925 ± 0.007 | 0.782 ± 0.022 |
| small | `grad_sum_lu_vjp` | f64 | `8x8` | 0.941 ± 0.012 | 0.766 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | `2x2` | 1.147 ± 0.064 | 0.461 ± 0.010 |
| small | `grad_sum_qr_jvp` | f64 | `4x4` | 0.937 ± 0.011 | 0.543 ± 0.008 |
| small | `grad_sum_qr_jvp` | f64 | `8x8` | 0.969 ± 0.108 | 0.454 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | `2x2` | 0.911 ± 0.008 | 0.748 ± 0.036 |
| small | `grad_sum_qr_vjp` | f64 | `4x4` | 0.882 ± 0.152 | 0.773 ± 0.110 |
| small | `grad_sum_qr_vjp` | f64 | `8x8` | 0.943 ± 0.004 | 0.807 ± 0.073 |
| small | `grad_sum_solve_jvp` | f64 | `2x2,rhs=1` | 1.102 ± 0.011 | 0.574 ± 0.013 |
| small | `grad_sum_solve_jvp` | f64 | `4x4,rhs=1` | 1.154 ± 0.040 | 0.580 ± 0.009 |
| small | `grad_sum_solve_jvp` | f64 | `8x8,rhs=1` | 1.374 ± 0.050 | 0.578 ± 0.011 |
| small | `grad_sum_solve_vjp` | f64 | `2x2,rhs=1` | 1.115 ± 0.189 | 0.711 ± 0.024 |
| small | `grad_sum_solve_vjp` | f64 | `4x4,rhs=1` | 1.113 ± 0.007 | 0.723 ± 0.021 |
| small | `grad_sum_solve_vjp` | f64 | `8x8,rhs=1` | 1.272 ± 0.021 | 0.708 ± 0.021 |
| small | `grad_sum_svd_s_jvp` | f64 | `2x2` | 0.621 ± 0.066 | 0.535 ± 0.088 |
| small | `grad_sum_svd_s_jvp` | f64 | `4x4` | 0.951 ± 0.003 | 0.592 ± 0.101 |
| small | `grad_sum_svd_s_jvp` | f64 | `8x8` | 0.838 ± 0.002 | 0.632 ± 0.009 |
| small | `grad_sum_svd_s_vjp` | f64 | `2x2` | 0.517 ± 0.013 | 0.451 ± 0.035 |
| small | `grad_sum_svd_s_vjp` | f64 | `4x4` | 0.932 ± 0.033 | 0.582 ± 0.056 |
| small | `grad_sum_svd_s_vjp` | f64 | `8x8` | 0.934 ± 0.011 | 0.564 ± 0.013 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A

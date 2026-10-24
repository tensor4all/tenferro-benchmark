# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/sparse`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261024_083043/gpu_sparse.yaml`
- Timestamp: `2026-10-24T08:30:43.977408+00:00`
- tenferro-rs commit: `06ac2475a286cdfd0332a2857db0bd78ba704aa5`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261024_083043/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-9f50e2ec-e8ed-3e7c-22f6-fb3e4ee47791`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/sparse / spmm / allocating output

| Problem | PyTorch CUDA |
|---|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | 14.868 |

## gpu/sparse / spmv / allocating output

| Problem | PyTorch CUDA |
|---|---:|
| sparse_synthetic_4m_64m_spmv_f64 | 2.121 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | 1 | 2.12078, 2.12233, 2.12174, 2.12027, 2.12043, 2.11923, 2.12503 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 1 | 14.7637, 14.7881, 14.8797, 14.878, 14.8787, 14.8676, 14.8632 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

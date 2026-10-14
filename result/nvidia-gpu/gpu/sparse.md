# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/sparse`
- Suite file: `/workspaces/tenferro-benchmark/.worktrees/gpu-vendors/data/results/nvidia-gpu/gpu/routine_selection/20261014_005047/gpu_sparse.yaml`
- Timestamp: `2026-10-14T00:50:47.476431+00:00`
- tenferro-rs commit: `9655b4e0daced17e0aa35a30d35710b069e10644`
- Coverage: `full`; effort: `scan`; manifest version: `1`
- Selection and case accounting: `/workspaces/tenferro-benchmark/.worktrees/gpu-vendors/data/results/nvidia-gpu/gpu/routine_selection/20261014_005047/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/sparse / spmm / allocating output

| Problem | PyTorch CUDA | PyTorch CSR (cuSPARSE) | Ginkgo |
|---|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | 108.275 | 108.289 | 1175.152 |

## gpu/sparse / spmv / allocating output

| Problem | PyTorch CUDA | PyTorch CSR (cuSPARSE) | Ginkgo |
|---|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | 19.705 | 19.703 | 20.906 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | 1 | 19.7072, 19.7054, 19.6997 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 1 | 108.275, 108.275, 108.267 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_4m_64m_spmv_f64 | cusparse | 1 | 19.7033, 19.7025, 19.7061 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | cusparse | 1 | 108.289, 108.275, 108.291 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_4m_64m_spmv_f64 | ginkgo | 1 | 20.9019, 20.9062, 20.9311 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | ginkgo | 1 | 1175.08, 1175.15, 1175.22 | inputs, layouts, native operators/workspaces, initialization and validation outside clocks; native float64; no dtype conversion in clocks |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | cusparse | 536870912 | 268435456 | 536870912 | 1 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | ginkgo | 536870912 | 268435456 | 536870912 | 1 |

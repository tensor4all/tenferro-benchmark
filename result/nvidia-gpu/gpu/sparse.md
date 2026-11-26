# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/sparse`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_sparse.yaml`
- Timestamp: `2026-11-26T06:29:50+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`
- Effort: `standard`; manifest version: `2`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd`
- Memory: `80 GiB`
- Driver version: `595.112.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC 7543 32-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `32`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,64-67; 4-7,68-71; 8-11,72-75; 12-15,76-79; 16-19,80-83; 20-23,84-87; 24-27,88-91; 28-31,92-95; 32-35,96-99; 36-39,100-103; 40-43,104-107; 44-47,108-111; 48-51,112-115; 52-55,116-119; 56-59,120-123; 60-63,124-127`
- CPU boost: `enabled`
- Container: `devcontainer`

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
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | 15.139 |

## gpu/sparse / spmv / allocating output

| Problem | PyTorch CUDA |
|---|---:|
| sparse_synthetic_4m_64m_spmv_f64 | 2.104 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | 1 | 2.10994, 2.11104, 2.10307, 2.10366, 2.10357, 2.10352, 2.10038 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 1 | 15.1486, 15.1349, 15.1382, 15.1418, 15.1421, 15.1386, 15.1356 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

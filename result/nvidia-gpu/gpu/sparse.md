# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/sparse`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/gpu_sparse.yaml`
- Timestamp: `2026-10-31T20:18:08.642468+00:00`
- tenferro-rs commit: `c15dfcf0ea5252effdef24cffbebf2f445916e20`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261031_201808/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-8840a2cd-10a3-a70c-f3a2-598e6aa3610d`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `256`
- Sockets: `2`
- Cores per socket: `64`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7,128-135; 8-15,136-143; 16-23,144-151; 24-31,152-159; 32-39,160-167; 40-47,168-175; 48-55,176-183; 56-63,184-191; 64-71,192-199; 72-79,200-207; 80-87,208-215; 88-95,216-223; 96-103,224-231; 104-111,232-239; 112-119,240-247; 120-127,248-255`
- CPU boost: `enabled`

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
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | 16.439 |

## gpu/sparse / spmv / allocating output

| Problem | PyTorch CUDA |
|---|---:|
| sparse_synthetic_4m_64m_spmv_f64 | 2.276 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | 1 | 2.27507, 2.28101, 2.2689, 2.27632, 2.27945, 2.27736, 2.27409 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 1 | 16.4449, 16.439, 16.4394, 16.4405, 16.4528, 14.4062, 14.39 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

## Output retention bounds

These intrinsic single outputs exceed the configured batch budget. Samples contain one operation; the effective bound is raised explicitly to that one output, without shrinking the workload.

| Problem | Backend | Bytes/output | Configured budget | Effective bound | Ops/sample |
|---|---|---:|---:|---:|---:|
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | 536870912 | 268435456 | 536870912 | 1 |

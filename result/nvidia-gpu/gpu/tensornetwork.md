# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/tensornetwork`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261017_141327/gpu_tensornetwork.yaml`
- Timestamp: `2026-10-17T14:13:27.525120+00:00`
- tenferro-rs commit: `4de2e1226337e877066f129667b70c2a01308bfb`
- Coverage: `full`; effort: `standard`; manifest version: `1`
- Selection and case accounting: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261017_141327/selection.json` (see adjacent `case_status.json` and `report.md`; unselected rows are not covered)

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-836a43ba-6520-b027-a26d-7c22feb348e8`
- Memory: `80 GiB`
- Driver version: `580.159.04`
- CUDA version: `13.0`
- CUDA runtime: `12.8`
- cuDNN version: `92000`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-124-generic-x86_64-with-glibc2.39`

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.

## gpu/tensornetwork / tensor_network_contract / allocating output

| Problem | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA |
|---|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | 53.519 | 51.005 | 110.586 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-trace | 1 | 54.2882, 54.1811, 54.3929, 53.6596, 53.3337, 53.3778, 53.7775, 53.259, 52.44, 51.8268 | one caller-held CUDA session entered before sampling; each Runtime::run_prepared_in call includes its own stream-retirement barrier inside the batch interval; inputs, plans, compilation and session entry outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | 1 | 51.5617, 51.3205, 51.9267, 50.9559, 50.7471, 52.1482, 50.9598, 51.0494, 49.9241, 49.9148 | one entered eager session; input wrapping, transfers, argument handles, initialization and validation outside clocks; native declared dtype; no AD recording |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | 1 | 109.609, 109.549, 110.514, 110.735, 111.39, 110.642, 110.121, 113.021, 110.594, 110.577 | inputs/transfers, plans, compilation, lazy initialization and validation outside clocks; JAX x64/highest; PyTorch TF32 disabled |

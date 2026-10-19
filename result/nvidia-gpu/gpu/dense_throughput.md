# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense_throughput`
- Suite file: `/workspaces/gpu-throughput/benchmarks/gpu/dense_throughput.yaml`
- Timestamp: `2026-10-19T09:28:49.176769+00:00`
- tenferro-rs commit: `e921b059a5e3c41ea76002553cbb9415d4b5e58d`

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

Median time is reported in milliseconds for `ok` records.
Inputs are prepared on the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization. tenferro-rs CUDA uses the explicit tenferro-rs synchronize API without downloading result tensors in the timed region.
Dense and einsum inputs use the same deterministic benchmark generator in the Rust and Python runners; host-to-device layout conversion remains outside the timed region.
tenferro-rs uses native column-major GPU tensors; PyTorch and vendor-wrapper columns use their native row-major framework tensors unless noted.
The cuSOLVER column is torch.linalg with preferred_linalg_library=cusolver; for SVD it pins driver=gesvd as a QR-based cuSOLVER comparison. tenferro-rs CUDA SVD uses its backend default driver policy, currently gesvdj for matrices with both dimensions at most 1024 and gesvd otherwise.
Non-`ok` cells show the structured backend status.
Rows whose backend name ends in `-deferred-batch` are **throughput mode**: each timed sample enqueues K deferred prepared runs (`Runtime::run_prepared_deferred`) in one caller-held session and pays one retirement barrier (`finish_prepared`), so the reported per-operation time is `sample_ms / K`. They measure batched throughput and are not comparable with, and never averaged together with, per-call latency rows.

## gpu/dense_throughput / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace (deferred batch throughput) |
|---|---:|
| dense_batched_matmul_f64_b16_32_throughput | 0.033 |

## gpu/dense_throughput / elementwise_chain / allocating output

| Problem | tenferro-rs CUDA trace (deferred batch throughput) |
|---|---:|
| elementwise_chain_f64_1k_throughput | 0.070 |
| elementwise_chain_f64_1m_throughput | 0.100 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_batched_matmul_f64_b16_32_throughput | tenferro-cuda-trace-deferred-batch | 64 | 2.10304, 2.08914, 2.1065, 2.08775, 2.06799, 2.08976, 2.04949 | one caller-held CUDA session entered before sampling; each timed sample enqueues K deferred prepared runs (Runtime::run_prepared_deferred) and pays one retirement barrier in a single finish_prepared; per-call time is sample_ms / K; inputs, plans, compilation, guard entry and session entry outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k_throughput | tenferro-cuda-trace-deferred-batch | 64 | 4.44892, 4.46153, 4.44963, 4.45863, 4.46369, 4.44391, 4.45345 | one caller-held CUDA session entered before sampling; each timed sample enqueues K deferred prepared runs (Runtime::run_prepared_deferred) and pays one retirement barrier in a single finish_prepared; per-call time is sample_ms / K; inputs, plans, compilation, guard entry and session entry outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m_throughput | tenferro-cuda-trace-deferred-batch | 32 | 3.19234, 3.1951, 3.18698, 3.18241, 3.18826, 3.18058, 3.19073 | one caller-held CUDA session entered before sampling; each timed sample enqueues K deferred prepared runs (Runtime::run_prepared_deferred) and pays one retirement barrier in a single finish_prepared; per-call time is sample_ms / K; inputs, plans, compilation, guard entry and session entry outside clocks; native declared dtype; no AD recording |

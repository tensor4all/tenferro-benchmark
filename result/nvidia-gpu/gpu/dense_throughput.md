# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense_throughput`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/benchmarks/gpu/dense_throughput.yaml`
- Timestamp: `2026-11-14T06:24:19.636034+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-3893cd38-00e0-11b0-a833-6c82285da154`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

## CPU Information

- Model: `AMD EPYC 7502 32-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `128`
- Sockets: `2`
- Cores per socket: `32`
- Threads per core: `2`
- NUMA nodes: `2`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
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
Rows whose backend name ends in `-deferred-batch` are **throughput mode**: each timed sample enqueues K deferred prepared runs (`Runtime::run_prepared_deferred`) in one caller-held session and pays one retirement barrier (`finish_prepared`), so the reported per-operation time is `sample_ms / K`. They measure batched throughput and are not comparable with, and never averaged together with, per-call latency rows.

## gpu/dense_throughput / batched_matmul / allocating output

| Problem | tenferro-rs CUDA trace (deferred batch throughput) |
|---|---:|
| dense_batched_matmul_f64_b16_32_throughput | 0.030 |

## gpu/dense_throughput / elementwise_chain / allocating output

| Problem | tenferro-rs CUDA trace (deferred batch throughput) |
|---|---:|
| elementwise_chain_f64_1k_throughput | 0.303 |
| elementwise_chain_f64_1m_throughput | 0.516 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_batched_matmul_f64_b16_32_throughput | tenferro-cuda-trace-deferred-batch | 64 | 1.94967, 1.95282, 1.94253, 1.96573, 1.92965, 1.93064, 1.92023 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k_throughput | tenferro-cuda-trace-deferred-batch | 64 | 19.5772, 19.6777, 19.5754, 19.3706, 19.2911, 19.3385, 19.3157 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m_throughput | tenferro-cuda-trace-deferred-batch | 32 | 16.5396, 16.7995, 16.4353, 16.5082, 16.4697, 16.434, 16.5173 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |

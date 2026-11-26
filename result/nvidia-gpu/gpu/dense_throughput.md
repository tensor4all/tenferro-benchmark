# GPU Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/dense_throughput`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/benchmarks/gpu/dense_throughput.yaml`
- Timestamp: `2026-11-26T06:57:55.851030+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-a95177dc-c682-83ff-0bc9-c1b3d7cca7ce`
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

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `252`
- Sockets: `2`
- Cores per socket: `126`
- Threads per core: `1`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-138-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-125; 126-251`
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
| dense_batched_matmul_f64_b16_32_throughput | 0.029 |

## gpu/dense_throughput / elementwise_chain / allocating output

| Problem | tenferro-rs CUDA trace (deferred batch throughput) |
|---|---:|
| elementwise_chain_f64_1k_throughput | 0.184 |
| elementwise_chain_f64_1m_throughput | 0.346 |

## Sampling

JAX uses compiled JIT execution; compilation, planning and numerical checks are outside timing. PyTorch uses eager execution. Samples retain every output until native synchronization and clock stop. Values are normalized per declared operation.

| Problem | Backend | Ops/sample | Raw batch ms | Setup and precision |
| --- | --- | ---: | --- | --- |
| dense_batched_matmul_f64_b16_32_throughput | tenferro-cuda-trace-deferred-batch | 64 | 1.8884, 1.86534, 1.90896, 1.86973, 1.92596, 1.8602, 1.84875 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1k_throughput | tenferro-cuda-trace-deferred-batch | 64 | 10.2255, 10.9056, 11.7648, 12.4402, 11.8598, 11.3159, 11.9212 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |
| elementwise_chain_f64_1m_throughput | tenferro-cuda-trace-deferred-batch | 32 | 11.1097, 11.0582, 11.0729, 11.1812, 11.0794, 11.0831, 11.1383 | one caller-held CUDA engine for every sample; each timed sample enqueues K prepared Runtime::run calls and pays one caller-side retirement barrier; per-call time is sample_ms / K; inputs, plans and compilation outside clocks; native declared dtype; no AD recording |

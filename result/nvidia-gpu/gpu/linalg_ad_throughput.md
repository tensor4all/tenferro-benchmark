# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_throughput`
- Suite file: `/workspaces/gpu-throughput/benchmarks/gpu/linalg_ad_throughput.yaml`
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

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned and checked against CPU F64 AD, whose directional derivative is checked by central differences. RHS and derivative seeds are prepared before timing.
Standard tenferro trace rows run the prepared graph in one caller-held CUDA session (Runtime::run_prepared_in); each timed call includes the run's own stream-retirement barrier. Small trace rows are separately labelled public-path latency diagnostics including internal session entry. Any unsupported row retains its actual reason in the raw records.
Rows whose backend name ends in `-deferred-batch` are **throughput mode**: each timed sample enqueues K deferred prepared runs (`Runtime::run_prepared_deferred`) in one caller-held CUDA session and pays one retirement barrier (`finish_prepared`), so the reported per-operation time is `sample_ms / K`. They measure batched throughput and are not comparable with, and never averaged together with, the per-call latency rows above.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.

## Linalg JVP/VJP Benchmark Items

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|
| large | `grad_sum_qr_jvp` | f64 | `256x256` | - | 1.426 ± 0.003 | - | - |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | - | 1.435 ± 0.004 | - | - |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-trace-deferred-batch | 64 | 7 | steady_state_batched_host_api_plus_one_retirement_per_k_samples |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-trace-deferred-batch | 64 | 7 | steady_state_batched_host_api_plus_one_retirement_per_k_samples |

## Loss Definitions

- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A

# GPU Linalg JVP/VJP Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/linalg_ad_throughput`
- Suite file: `/tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/benchmarks/gpu/linalg_ad_throughput.yaml`
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

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext` on CUDA; PyTorch uses `torch.func.jvp` / `vjp` on CUDA.
JAX uses an explicitly lowered/compiled whole-operation JIT; compilation and lazy initialization occur before sampling. Primal loss and first-order derivatives with respect to A are both returned; the check that backs each row is stated below, and a row that carries none is marked as not checked. RHS and derivative seeds are prepared before timing.
Standard and latency AD rows time one prepared operation plus device completion per call on every backend, repeated in bounded batches. tenferro uses Runtime::run on a reusable caller-held CUDA engine. Fixtures, transfers, graph construction, compilation, preparation and initialization are untimed; output conversion and verification follow the stopped clock. Small rows measure API latency, not kernel throughput. Legacy session-entry measurements require recollection (issue #309). Any unsupported row retains its actual reason in the raw records.
Rows whose backend name ends in `-deferred-batch` are **throughput mode**: each timed sample enqueues K deferred prepared runs (`Runtime::run`) on one caller-held CUDA engine and pays one native CUDA synchronization barrier, so the reported per-operation time is `sample_ms / K`. They measure batched throughput and are not comparable with, and never averaged together with, the per-call latency rows above.
Inputs are uploaded to the GPU before timed runs; initial host-to-device transfer is outside the timed region.
Timed runs include the host API call and backend-native device synchronization without downloading AD outputs in the timed region.
Verification modes: `cpu_reference` (re-executed on the CPU in f64 and compared element by element); `finite_difference` (directional derivative compared with a central difference of the primal loss at a step derived from f64 precision). Every mode is held to the suite's own rtol/atol.

## Linalg JVP/VJP Benchmark Items

### Throughput regime

Compute-bound rows: every case here is over its family's floor in `docs/regime-floors.md`. They are never averaged with, or read as a ratio against, the other section.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_qr_jvp` | f64 | `4096x4096` | - | - | 148.220 ± 0.231 | - | - |

### Latency regime

Per-call latency rows: the device kernels are a small fraction of each number, so these are not GPU throughput.

| suite | benchmark | dtype | shape | tenferro-rs CUDA trace | tenferro-rs CUDA eager, per call, incl. AD prepare | tenferro-rs CUDA trace (deferred batch throughput) | PyTorch CUDA | JAX CUDA JIT |
|---|---|---:|---|---:|---:|---:|---:|---:|
| large | `grad_sum_qr_jvp` | f64 | `256x256` | - | - | 2.786 ± 0.009 | - | - |
| large | `grad_sum_qr_vjp` | f64 | `256x256` | - | - | 2.773 ± 0.006 | - | - |

## Sampling and unsupported routes

| Case | Backend | Ops/sample | Samples | Scope / reason |
| --- | --- | ---: | ---: | --- |
| linalg_ad_large_grad_sum_qr_jvp_f64_256 | tenferro-cuda-trace-deferred-batch | 64 | 7 | steady_state_batched_host_api_plus_one_retirement_per_k_samples |
| linalg_ad_large_grad_sum_qr_vjp_f64_256 | tenferro-cuda-trace-deferred-batch | 64 | 7 | steady_state_batched_host_api_plus_one_retirement_per_k_samples |
| linalg_ad_large_grad_sum_qr_jvp_f64_4096 | tenferro-cuda-trace-deferred-batch | 64 | 7 | steady_state_batched_host_api_plus_one_retirement_per_k_samples |

## Loss Definitions

- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A

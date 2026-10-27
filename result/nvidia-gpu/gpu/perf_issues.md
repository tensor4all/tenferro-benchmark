# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261027_012236`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-18e564dc-3f3c-b8d2-bcdf-5455b61dd732`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`

## run_t1

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261027_012236/run_t1.yaml`

```yaml
timestamp: '2026-10-27T01:23:06Z'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 234a51369c257ffd311e9e547b75b394b4adb5dc
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: none
environment:
  hostname: 2da32d4129f4
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: da438501c84c802eb0b4eb4f420a181ef52ce38b
    OPENBLAS_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: 18cbfccb7b902aeac7e8cfc9a0bf3588b46c56a5dedcf9fb16673f76d7bb546b
    GPU_BENCH_CLOCK_RAMP_MS: '300'
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    GPU_BENCH_PREBUILT_RUST: '1'
    BENCHMARK_EXECUTION_ENVIRONMENT: runpod
    BENCH_INSTANCE: solve_readback_latency_f64_n2,solve_readback_latency_f64_n2_pytorch,solve_readback_latency_f64_n4,solve_readback_latency_f64_n4_pytorch,solve_readback_latency_f64_n8,solve_readback_latency_f64_n8_pytorch,solve_readback_latency_f64_n32,solve_readback_latency_f64_n32_pytorch,qr_readback_latency_f64_n2,qr_readback_latency_f64_n2_pytorch,qr_readback_latency_f64_n4,qr_readback_latency_f64_n4_pytorch,qr_readback_latency_f64_n8,qr_readback_latency_f64_n8_pytorch,qr_readback_latency_f64_n32,qr_readback_latency_f64_n32_pytorch
    CUDA_VISIBLE_DEVICES: GPU-18e564dc-3f3c-b8d2-bcdf-5455b61dd732
    GPU_BENCH_PHYSICAL_DEVICE: '0'
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 8, coverage `full`, effort `standard`, filter `solve_readback_latency_f64_n2,solve_readback_latency_f64_n2_pytorch,solve_readback_latency_f64_n4,solve_readback_latency_f64_n4_pytorch,solve_readback_latency_f64_n8,solve_readback_latency_f64_n8_pytorch,solve_readback_latency_f64_n32,solve_readback_latency_f64_n32_pytorch,qr_readback_latency_f64_n2,qr_readback_latency_f64_n2_pytorch,qr_readback_latency_f64_n4,qr_readback_latency_f64_n4_pytorch,qr_readback_latency_f64_n8,qr_readback_latency_f64_n8_pytorch,qr_readback_latency_f64_n32,qr_readback_latency_f64_n32_pytorch`
- Expected 200, selected 16, executed 16, unsupported 0, failed 0, missing 0, noisy 0.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Not selected: `transfer_up_f64_tenferro_8B@t1`, `transfer_up_f64_cudarc-pageable_8B@t1`, `transfer_up_f64_cudarc-pinned_8B@t1`, `transfer_up_f64_tenferro_4096B@t1`, `transfer_up_f64_cudarc-pageable_4096B@t1`, `transfer_up_f64_cudarc-pinned_4096B@t1`, `transfer_up_f64_tenferro_65536B@t1`, `transfer_up_f64_cudarc-pageable_65536B@t1`, `transfer_up_f64_cudarc-pinned_65536B@t1`, `transfer_up_f64_tenferro_1048576B@t1`, `transfer_up_f64_cudarc-pageable_1048576B@t1`, `transfer_up_f64_cudarc-pinned_1048576B@t1`, `transfer_up_f64_tenferro_16777216B@t1`, `transfer_up_f64_cudarc-pageable_16777216B@t1`, `transfer_up_f64_cudarc-pinned_16777216B@t1`, `transfer_up_f64_tenferro_268435456B@t1`, `transfer_up_f64_cudarc-pageable_268435456B@t1`, `transfer_up_f64_cudarc-pinned_268435456B@t1`, `transfer_down_f64_tenferro_8B@t1`, `transfer_down_f64_cudarc-pageable_8B@t1`, `transfer_down_f64_cudarc-pinned_8B@t1`, `transfer_down_f64_tenferro-into_8B@t1`, `transfer_down_f64_cudarc-pageable-fresh_8B@t1`, `transfer_down_f64_tenferro_4096B@t1`, `transfer_down_f64_cudarc-pageable_4096B@t1`, `transfer_down_f64_cudarc-pinned_4096B@t1`, `transfer_down_f64_tenferro-into_4096B@t1`, `transfer_down_f64_cudarc-pageable-fresh_4096B@t1`, `transfer_down_f64_tenferro_65536B@t1`, `transfer_down_f64_cudarc-pageable_65536B@t1`, `transfer_down_f64_cudarc-pinned_65536B@t1`, `transfer_down_f64_tenferro-into_65536B@t1`, `transfer_down_f64_cudarc-pageable-fresh_65536B@t1`, `transfer_down_f64_tenferro_1048576B@t1`, `transfer_down_f64_cudarc-pageable_1048576B@t1`, `transfer_down_f64_cudarc-pinned_1048576B@t1`, `transfer_down_f64_tenferro-into_1048576B@t1`, `transfer_down_f64_cudarc-pageable-fresh_1048576B@t1`, `transfer_down_f64_tenferro_16777216B@t1`, `transfer_down_f64_cudarc-pageable_16777216B@t1` (+144 more)

Numerical checks: **16/16 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| solve_readback_latency_f64_n2 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=2 | 1 | 8 | 434053.8 | 13552.0 | 4.1% | ok |
| solve_readback_latency_f64_n2_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=2 | 1 | 32 | 112034.4 | 1667.8 | 2.7% | ok |
| solve_readback_latency_f64_n4 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=4 | 1 | 8 | 278188.5 | 4931.8 | 1.3% | ok |
| solve_readback_latency_f64_n4_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=4 | 1 | 16 | 112821.6 | 1848.2 | 5.1% | ok |
| solve_readback_latency_f64_n8 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=8 | 1 | 8 | 257817.6 | 11521.6 | 7.1% | ok |
| solve_readback_latency_f64_n8_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=8 | 1 | 32 | 112420.5 | 1371.0 | 2.1% | ok |
| solve_readback_latency_f64_n32 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=32 | 1 | 8 | 277950.5 | 4195.5 | 1.3% | ok |
| solve_readback_latency_f64_n32_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=32 | 1 | 16 | 141797.5 | 883.6 | 1.6% | ok |
| qr_readback_latency_f64_n2 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=2 | 1 | 8 | 278204.8 | 4781.5 | 1.3% | ok |
| qr_readback_latency_f64_n2_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=2 | 1 | 16 | 139361.6 | 898.9 | 3.3% | ok |
| qr_readback_latency_f64_n4 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=4 | 1 | 8 | 279392.0 | 19531.8 | 5.4% | ok |
| qr_readback_latency_f64_n4_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=4 | 1 | 32 | 90128.8 | 7731.5 | 7.4% | ok |
| qr_readback_latency_f64_n8 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=8 | 1 | 8 | 278828.6 | 13217.5 | 4.6% | ok |
| qr_readback_latency_f64_n8_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=8 | 1 | 32 | 86633.1 | 1144.5 | 1.3% | ok |
| qr_readback_latency_f64_n32 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=32 | 1 | 8 | 282210.0 | 12842.4 | 3.0% | ok |
| qr_readback_latency_f64_n32_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=32 | 1 | 16 | 147410.6 | 179.7 | 0.2% | ok |


## Timing boundaries

- **linalg_readback_latency / direct** — inside: one CUDA solve or QR of an n x n f64 device matrix, dispatch and batch-edge device sync, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, residual or reconstruction check (download).
- **linalg_readback_latency / pytorch** — inside: one public PyTorch CUDA operation, batch-edge device synchronization; outside: fixture, upload and layout conversion, lazy initialization, validation and download, retention allocation and output destruction.

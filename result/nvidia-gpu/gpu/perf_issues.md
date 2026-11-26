# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261126_104707`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-8cf6aabc-9af9-c836-00eb-bd6ea7271174`
- Memory: `80 GiB`
- Driver version: `595.112.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`
- Container: `devcontainer`

## run_t1

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261126_104707/run_t1.yaml`

```yaml
timestamp: '2026-11-26T10:47:10Z'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: devcontainer
  hostname: 590b08b58fca
  os: Linux-6.8.0-149-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: null
    measured_cpus: 0-127
    l3_domains:
    - 0-3
    - 4-7
    - 8-11
    - 12-15
    - 16-19
    - 20-23
    - 24-27
    - 28-31
    - 32-35
    - 36-39
    - 40-43
    - 44-47
    - 48-51
    - 52-55
    - 56-59
    - 60-63
    - 64-67
    - 68-71
    - 72-75
    - 76-79
    - 80-83
    - 84-87
    - 88-91
    - 92-95
    - 96-99
    - 100-103
    - 104-107
    - 108-111
    - 112-115
    - 116-119
    - 120-123
    - 124-127
    cpu_boost: enabled
    scaling_governor: performance
    scaling_driver: acpi-cpufreq
  env:
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: d95f6d66c3699032bcd1154757a615fb69b3a21c
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: a261fcdef53ecfd3885b6d565dc834faef59287ce2da88c3d6754be96741f84d
    GPU_BENCH_CLOCK_RAMP_MS: '300'
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    GPU_BENCH_PREBUILT_RUST: '1'
    BENCHMARK_EXECUTION_ENVIRONMENT: runpod
    BENCH_INSTANCE: null
    CUDA_VISIBLE_DEVICES: GPU-8cf6aabc-9af9-c836-00eb-bd6ea7271174
    GPU_BENCH_PHYSICAL_DEVICE: '0'
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 12, coverage `full`, effort `standard`
- Expected 198, selected 198, executed 198, unsupported 0, failed 0, skipped 0, missing 0, noisy 15.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 198 | 198 | 198 | 0 | 0 | 0 | 0 | 15 |

- Noisy: `transfer_up_f64_cudarc-pageable_16777216B@t1`, `transfer_up_f64_tenferro_268435456B@t1`, `transfer_down_f64_tenferro_268435456B@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`, `qr_live_buffers_f64_n32_live256@t1`, `eager_op_f32_dot_general_decode_eager-session@t1`, `vendor_handle_cublas_f64_create-destroy@t1`, `eigh_batched_f64_n16_divide-and-conquer@t1`, `eigh_batched_f64_n32_divide-and-conquer@t1`, `eigh_batched_f64_n128_auto@t1`, `eigh_batched_f64_n128_divide-and-conquer@t1`, `eigh_batched_c64_n16_auto@t1`, `eigh_batched_c64_n32_auto@t1`, `eigh_batched_c64_n32_divide-and-conquer@t1`, `contraction_plan_churn_f64_repeated-plan@t1`

Numerical checks: **198/198 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| cholesky_readback_latency_f64_n512 | #2436 #2440 #2445 | direct | tenferro-cuda | f64 | n=512 | 1 | 8 | 333546.6 | 797.8 | 0.2% | ok |
| cholesky_readback_latency_f64_n512_eager | #2436 #2440 #2445 | eager | tenferro-cuda-eager | f64 | n=512 | 1 | 8 | 355606.0 | 1822.9 | 0.7% | ok |
| cholesky_readback_latency_f64_n512_pytorch | #2436 #2440 #2445 | pytorch | pytorch-cuda | f64 | n=512 | 1 | 8 | 336618.8 | 1281.9 | 0.4% | ok |
| cholesky_readback_latency_f32_n512 | #2436 #2440 #2445 | direct | tenferro-cuda | f32 | n=512 | 1 | 8 | 274355.2 | 499.6 | 0.2% | ok |
| cholesky_readback_latency_f32_n512_eager | #2436 #2440 #2445 | eager | tenferro-cuda-eager | f32 | n=512 | 1 | 8 | 304147.0 | 4404.7 | 0.8% | ok |
| cholesky_readback_latency_f32_n512_pytorch | #2436 #2440 #2445 | pytorch | pytorch-cuda | f32 | n=512 | 1 | 8 | 276768.8 | 679.4 | 0.3% | ok |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 16179.9 | 91.5 | 0.4% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 10779.9 | 154.7 | 1.0% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 14890.2 | 132.3 | 0.7% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 16062.9 | 103.0 | 1.3% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 256 | 11109.2 | 29.9 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 32 | 63613.1 | 3037.7 | 4.3% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 128 | 20660.3 | 286.4 | 2.7% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 128 | 18998.2 | 124.4 | 0.7% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 128 | 18180.3 | 63.2 | 0.6% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 32 | 90473.9 | 412.0 | 0.2% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 101094.1 | 640.4 | 0.8% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 57342.4 | 372.0 | 0.7% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 1 | 2157351.0 | 46382.5 | 1.6% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 2 | 1043288.5 | 47009.2 | 12.3% | NOISY |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 673943.8 | 10368.4 | 0.9% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 184664021.0 | 53513628.5 | 19.2% | NOISY |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 17056486.0 | 120123.5 | 0.9% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10275504.0 | 21731.0 | 0.1% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 15920.0 | 144.8 | 1.2% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 14223.5 | 95.1 | 0.7% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 128 | 15033.2 | 91.9 | 0.7% | ok |
| transfer_down_f64_tenferro-into_8B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 256 | 15151.7 | 579.4 | 2.9% | ok |
| transfer_down_f64_cudarc-pageable-fresh_8B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 11605.1 | 52.6 | 2.9% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 17531.0 | 127.3 | 3.2% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 128 | 15083.6 | 205.9 | 4.3% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 128 | 16371.0 | 549.0 | 3.0% | ok |
| transfer_down_f64_tenferro-into_4096B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 256 | 15229.8 | 41.1 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable-fresh_4096B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 12073.2 | 57.3 | 0.4% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 42882.8 | 1954.0 | 4.8% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 128 | 22907.9 | 82.7 | 0.4% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 128 | 16206.5 | 145.6 | 0.6% | ok |
| transfer_down_f64_tenferro-into_65536B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 128 | 18577.3 | 212.2 | 3.4% | ok |
| transfer_down_f64_cudarc-pageable-fresh_65536B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=65536, direction=down | 1 | 64 | 40846.9 | 1784.4 | 4.2% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 441093.5 | 4328.9 | 1.6% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 114869.9 | 510.2 | 2.2% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 57339.5 | 452.7 | 0.5% | ok |
| transfer_down_f64_tenferro-into_1048576B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 64 | 53849.7 | 767.3 | 1.7% | ok |
| transfer_down_f64_cudarc-pageable-fresh_1048576B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=1048576, direction=down | 1 | 8 | 474088.8 | 18509.6 | 6.5% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 1 | 3918168.0 | 60706.0 | 1.4% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 1816373.5 | 144809.8 | 7.5% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 1131130.5 | 18112.0 | 1.9% | ok |
| transfer_down_f64_tenferro-into_16777216B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 1159259.0 | 4338.2 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable-fresh_16777216B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=16777216, direction=down | 1 | 1 | 4320343.0 | 261737.0 | 4.7% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 362040075.0 | 47637036.5 | 11.4% | NOISY |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 27064853.0 | 468024.5 | 1.7% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 18706192.0 | 208602.0 | 1.5% | ok |
| transfer_down_f64_tenferro-into_268435456B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 20011333.0 | 312685.0 | 1.2% | ok |
| transfer_down_f64_cudarc-pageable-fresh_268435456B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 105144678.0 | 1433735.0 | 1.6% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 4096 | 886.7 | 24.2 | 3.6% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 4096 | 976.0 | 2.6 | 1.6% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 1024 | 3272.3 | 52.4 | 1.5% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 256 | 8965.6 | 96.2 | 2.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 512 | 3887.9 | 10.1 | 1.4% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 256 | 8645.0 | 163.2 | 2.0% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 512 | 4467.6 | 54.8 | 3.4% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 32 | 20054.4 | 4204.7 | 19.0% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 11511.6 | 187.0 | 1.2% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1729182.5 | 36827.5 | 1.6% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 3244408.0 | 36153.0 | 2.9% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 16943.8 | 12.6 | 0.1% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 3345370.0 | 23600.5 | 0.6% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 14658.3 | 57.8 | 0.5% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 3537094.0 | 9949.0 | 0.3% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 14255.9 | 51.5 | 0.7% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 4233669.0 | 7284.0 | 0.4% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 4 | 34067.2 | 717.6 | 2.6% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2913138.0 | 4002.5 | 0.4% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2262902.0 | 1168.0 | 0.1% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 47083509.0 | 248598.0 | 0.3% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35227730.0 | 90181.5 | 0.2% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 15990710.0 | 211427.0 | 1.0% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14798041.0 | 50711.5 | 0.5% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 253511313.0 | 450546.5 | 0.2% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 231721860.0 | 348928.5 | 0.1% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 16 | 182202.0 | 7209.9 | 2.7% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 16 | 181658.4 | 286.2 | 43.2% | NOISY |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 16 | 184535.8 | 4722.8 | 2.3% | ok |
| prepared_dot_general_f32_decode_len8 | #2197 | prepared | tenferro-cuda-trace | f32 | in=1024, out=1024, len=8 | 1 | 64 | 38302.0 | 359.6 | 1.9% | ok |
| prepared_dot_general_f32_decode_len8_eager-session | #2197 | eager-session | tenferro-cuda-eager | f32 | in=1024, out=1024, len=8 | 1 | 128 | 19560.0 | 239.4 | 3.0% | ok |
| prepared_dot_general_f32_decode_len8_pytorch | #2197 | pytorch | pytorch-cuda | f32 | in=1024, out=1024, len=8 | 1 | 128 | 15784.4 | 10.6 | 0.1% | ok |
| standard_batched_matmul_f64_b16_32_cublas | #2193 | batch-trailing | tenferro-cuda | f64 | n=32, batch=16 | 1 | 128 | 13666.3 | 124.4 | 0.7% | ok |
| standard_batched_matmul_f64_b16_32_cublas_batch-outer | #2193 | batch-outer | tenferro-cuda | f64 | n=32, batch=16 | 1 | 128 | 21248.7 | 70.1 | 2.7% | ok |
| standard_batched_matmul_f64_b16_32_cublas_pytorch | #2193 | pytorch | pytorch-cuda | f64 | n=32, batch=16 | 1 | 128 | 12113.6 | 204.2 | 4.7% | ok |
| solve_readback_latency_f64_n2 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=2 | 1 | 16 | 125007.6 | 325.9 | 0.4% | ok |
| solve_readback_latency_f64_n2_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=2 | 1 | 16 | 124089.6 | 260.5 | 0.2% | ok |
| solve_readback_latency_f64_n4 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=4 | 1 | 16 | 123491.0 | 368.9 | 0.3% | ok |
| solve_readback_latency_f64_n4_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=4 | 1 | 16 | 126661.4 | 1382.3 | 3.9% | ok |
| solve_readback_latency_f64_n8 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=8 | 1 | 16 | 125343.9 | 731.7 | 3.0% | ok |
| solve_readback_latency_f64_n8_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=8 | 1 | 16 | 126500.5 | 612.4 | 0.4% | ok |
| solve_readback_latency_f64_n32 | #2192 #2204 | direct | tenferro-cuda | f64 | op=solve, n=32 | 1 | 16 | 140066.9 | 361.3 | 0.2% | ok |
| solve_readback_latency_f64_n32_pytorch | #2192 #2204 | pytorch | pytorch-cuda | f64 | op=solve, n=32 | 1 | 16 | 145774.1 | 624.6 | 0.3% | ok |
| qr_readback_latency_f64_n2 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=2 | 1 | 16 | 126739.7 | 624.9 | 0.5% | ok |
| qr_readback_latency_f64_n2_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=2 | 1 | 32 | 101606.9 | 401.5 | 0.3% | ok |
| qr_readback_latency_f64_n4 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=4 | 1 | 16 | 123670.8 | 213.6 | 0.1% | ok |
| qr_readback_latency_f64_n4_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=4 | 1 | 32 | 102141.7 | 923.0 | 2.2% | ok |
| qr_readback_latency_f64_n8 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=8 | 1 | 16 | 125923.1 | 225.8 | 0.4% | ok |
| qr_readback_latency_f64_n8_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=8 | 1 | 32 | 102168.6 | 699.5 | 2.1% | ok |
| qr_readback_latency_f64_n32 | #2195 | direct | tenferro-cuda | f64 | op=qr, n=32 | 1 | 16 | 186663.0 | 941.2 | 0.5% | ok |
| qr_readback_latency_f64_n32_pytorch | #2195 | pytorch | pytorch-cuda | f64 | op=qr, n=32 | 1 | 16 | 148745.2 | 81.4 | 1.0% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 7158.1 | 202.5 | 2.6% | ok |
| eager_op_f32_add_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 8116.1 | 33.7 | 0.8% | ok |
| eager_op_f32_add_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 4175.5 | 59.8 | 2.1% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 20321.3 | 1711.6 | 40.9% | NOISY |
| eager_op_f32_dot_general_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 21450.8 | 152.0 | 0.5% | ok |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 256 | 12075.7 | 73.1 | 0.9% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 64 | 39109.8 | 452.6 | 1.2% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 256 | 12203.3 | 328.5 | 1.8% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 64 | 37666.5 | 601.4 | 1.2% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 256 | 11712.4 | 119.2 | 2.4% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 64 | 37889.8 | 895.0 | 2.9% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 386442.2 | 155.1 | 0.1% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 496133.2 | 2296.9 | 0.3% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 #1720 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4620634.0 | 6262.0 | 0.1% | ok |
| svd_driver_c64_n64_jacobi | #1839 #1849 #1720 | jacobi | tenferro-cuda | c64 | n=64, driver=jacobi | 1 | 1 | 4638537.0 | 2935.5 | 0.1% | ok |
| svd_driver_c64_n64_qr-based | #1839 #1849 #1720 | qr-based | tenferro-cuda | c64 | n=64, driver=qr-based | 1 | 1 | 6942287.0 | 83193.0 | 0.7% | ok |
| svd_driver_c64_n64_polar | #1839 #1849 #1720 | polar | tenferro-cuda | c64 | n=64, driver=polar | 1 | 1 | 3440070.0 | 12203.0 | 0.3% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 #1720 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41110058.0 | 23619.5 | 0.1% | ok |
| svd_driver_c64_n256_jacobi | #1839 #1849 #1720 | jacobi | tenferro-cuda | c64 | n=256, driver=jacobi | 1 | 1 | 41078138.0 | 25989.0 | 0.7% | ok |
| svd_driver_c64_n256_qr-based | #1839 #1849 #1720 | qr-based | tenferro-cuda | c64 | n=256, driver=qr-based | 1 | 1 | 33374704.0 | 420610.5 | 0.8% | ok |
| svd_driver_c64_n256_polar | #1839 #1849 #1720 | polar | tenferro-cuda | c64 | n=256, driver=polar | 1 | 1 | 9303055.0 | 37741.5 | 0.7% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 #1720 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 388860037.0 | 457259.5 | 0.1% | ok |
| svd_driver_c64_n1024_jacobi | #1839 #1849 #1720 | jacobi | tenferro-cuda | c64 | n=1024, driver=jacobi | 1 | 1 | 387975376.0 | 287441.5 | 0.1% | ok |
| svd_driver_c64_n1024_qr-based | #1839 #1849 #1720 | qr-based | tenferro-cuda | c64 | n=1024, driver=qr-based | 1 | 1 | 267889497.0 | 472152.5 | 0.2% | ok |
| svd_driver_c64_n1024_polar | #1839 #1849 #1720 | polar | tenferro-cuda | c64 | n=1024, driver=polar | 1 | 1 | 61655104.0 | 95251.5 | 0.2% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 258575.2 | 1176.0 | 0.4% | ok |
| eigh_batched_f64_n16_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | f64 | n=16, batch=64, driver=divide-and-conquer | 1 | 2 | 264142.0 | 4345.8 | 28.7% | NOISY |
| eigh_batched_f64_n16_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | f64 | n=16, batch=64, driver=jacobi | 1 | 16 | 143423.2 | 872.9 | 0.4% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 397917.6 | 2059.4 | 0.4% | ok |
| eigh_batched_f64_n32_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | f64 | n=32, batch=64, driver=divide-and-conquer | 1 | 2 | 403763.0 | 45797.0 | 79.9% | NOISY |
| eigh_batched_f64_n32_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | f64 | n=32, batch=64, driver=jacobi | 1 | 8 | 324578.2 | 1744.6 | 0.5% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 2 | 5655927.0 | 6552857.0 | 65.0% | NOISY |
| eigh_batched_f64_n128_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | f64 | n=128, batch=64, driver=divide-and-conquer | 1 | 2 | 3254737.0 | 3252638.2 | 64.5% | NOISY |
| eigh_batched_f64_n128_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | f64 | n=128, batch=64, driver=jacobi | 1 | 1 | 20167458.0 | 12048.0 | 0.1% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 2 | 305330.5 | 47039.2 | 18.1% | NOISY |
| eigh_batched_c64_n16_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | c64 | n=16, batch=64, driver=divide-and-conquer | 1 | 2 | 309002.5 | 4699.2 | 1.5% | ok |
| eigh_batched_c64_n16_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | c64 | n=16, batch=64, driver=jacobi | 1 | 16 | 201016.5 | 1457.8 | 0.4% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 4 | 460410.0 | 227029.0 | 56.4% | NOISY |
| eigh_batched_c64_n32_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | c64 | n=32, batch=64, driver=divide-and-conquer | 1 | 2 | 814814.0 | 578820.5 | 86.8% | NOISY |
| eigh_batched_c64_n32_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | c64 | n=32, batch=64, driver=jacobi | 1 | 4 | 563593.8 | 1076.9 | 0.2% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3305594.0 | 28229.0 | 1.0% | ok |
| eigh_batched_c64_n128_divide-and-conquer | #1846 #1852 #1923 | divide-and-conquer | tenferro-cuda | c64 | n=128, batch=64, driver=divide-and-conquer | 1 | 1 | 3233016.0 | 6552.0 | 0.2% | ok |
| eigh_batched_c64_n128_jacobi | #1846 #1852 #1923 | jacobi | tenferro-cuda | c64 | n=128, batch=64, driver=jacobi | 1 | 1 | 49451365.0 | 154609.0 | 1.0% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 1 | 14626545.0 | 30563.5 | 0.1% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 7495.3 | 27.7 | 0.2% | ok |


## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 2048 | 910.9 | 6.9 | 0.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 512 | 3771.6 | 23.6 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 512 | 4173.2 | 15.7 | 0.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 256 | 4798.8 | 33.5 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 11384.8 | 161.0 | 0.9% | ok |
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 16384 | 230.2 | 2.3 | 1.9% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 2 | 255.5 | 5.0 | 6.5% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 31069154.0 | 9307.5 | 0.0% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 7429.2 | 75.9 | 1.3% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 19661.0 | 99.8 | 0.5% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 2 | 1409905.5 | 565104.2 | 33.6% | NOISY |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 25027.3 | 323.8 | 1.1% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so a sample is exactly one call: the median is that call's latency, not an average over a batch, and the precondition sits outside its clock.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 60886.0 | 496.0 | 0.7% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 30188.0 | 1036.5 | 18.7% | NOISY |
| contraction_plan_churn_f64_below-capacity-varying-workspace | #1873 #1805 | below-capacity-varying-workspace | tenferro-cuda | f64 | distinct=48, n=64 | 1 | 1 | 53392.0 | 485.5 | 0.7% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 42781.0 | 240.0 | 0.7% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 65024.0 | 3617.5 | 4.1% | ok |
| borrowed_input_gemm_f64_n8_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 26871.0 | 1238.5 | 3.8% | ok |
| borrowed_input_gemm_f64_n8_view-input | #1925 | view-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 37141.0 | 2349.0 | 8.9% | ok |
| borrowed_input_gemm_f64_n32_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 24877.0 | 526.0 | 4.2% | ok |
| borrowed_input_gemm_f64_n32_view-input | #1925 | view-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 37431.0 | 1112.0 | 1.8% | ok |

## Whole-sequence CUDA shape-stream latency

One sample is the entire recorded sequence, including required planning and one final native synchronization. Cache reset and explicitly declared key warmup are outside timing. Median/IQR describe sequence latency, not single-operation throughput. Enqueue durations and the exact key/shape/stride sequence are retained in raw rows. Counter observations run separately. NOISY is descriptive; these screens establish no baseline/candidate performance claim.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Sequences/sample | Median ns/sequence | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| shape_stream_f64_n16_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 415590.0 | 10936.0 | 4.7% | ok |
| shape_stream_f64_n16_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 420039.0 | 5189.5 | 1.2% | ok |
| shape_stream_f64_n16_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 134576.0 | 7388.5 | 6.3% | ok |
| shape_stream_f64_n16_capacity-thrash_latency | #1873 #1805 | capacity-thrash | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 3279845.0 | 15445.5 | 0.4% | ok |
| shape_stream_f64_n16_equal-shape-strides_latency | #1873 #1805 | equal-shape-strides | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 683549.0 | 22643.5 | 2.3% | ok |
| shape_stream_f64_n64_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 426871.0 | 8190.5 | 2.8% | ok |
| shape_stream_f64_n64_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 424978.0 | 9202.5 | 1.5% | ok |
| shape_stream_f64_n64_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 127903.0 | 3421.5 | 3.6% | ok |
| shape_stream_f64_n64_capacity-thrash_latency | #1873 #1805 | capacity-thrash | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 3399212.0 | 170959.5 | 8.8% | ok |
| shape_stream_f64_n64_equal-shape-strides_latency | #1873 #1805 | equal-shape-strides | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 637382.0 | 5876.5 | 0.9% | ok |

### shape_stream_f64_n16_fixed-warm_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]}]
```


### shape_stream_f64_n16_mixed-warm_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]}]
```


### shape_stream_f64_n16_fresh_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":3,"lhs_layout":"compact","lhs_shape":[19,2,16],"lhs_strides":[1,19,38],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":4,"lhs_layout":"compact","lhs_shape":[20,2,16],"lhs_strides":[1,20,40],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":5,"lhs_layout":"compact","lhs_shape":[21,2,16],"lhs_strides":[1,21,42],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":6,"lhs_layout":"compact","lhs_shape":[22,2,16],"lhs_strides":[1,22,44],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":7,"lhs_layout":"compact","lhs_shape":[23,2,16],"lhs_strides":[1,23,46],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]}]
```


### shape_stream_f64_n16_capacity-thrash_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"compact","lhs_shape":[17,2,16],"lhs_strides":[1,17,34],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":2,"lhs_layout":"compact","lhs_shape":[18,2,16],"lhs_strides":[1,18,36],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":3,"lhs_layout":"compact","lhs_shape":[19,2,16],"lhs_strides":[1,19,38],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":4,"lhs_layout":"compact","lhs_shape":[20,2,16],"lhs_strides":[1,20,40],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":5,"lhs_layout":"compact","lhs_shape":[21,2,16],"lhs_strides":[1,21,42],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":6,"lhs_layout":"compact","lhs_shape":[22,2,16],"lhs_strides":[1,22,44],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":7,"lhs_layout":"compact","lhs_shape":[23,2,16],"lhs_strides":[1,23,46],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":8,"lhs_layout":"compact","lhs_shape":[24,2,16],"lhs_strides":[1,24,48],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":9,"lhs_layout":"compact","lhs_shape":[25,2,16],"lhs_strides":[1,25,50],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":10,"lhs_layout":"compact","lhs_shape":[26,2,16],"lhs_strides":[1,26,52],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":11,"lhs_layout":"compact","lhs_shape":[27,2,16],"lhs_strides":[1,27,54],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":12,"lhs_layout":"compact","lhs_shape":[28,2,16],"lhs_strides":[1,28,56],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":13,"lhs_layout":"compact","lhs_shape":[29,2,16],"lhs_strides":[1,29,58],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":14,"lhs_layout":"compact","lhs_shape":[30,2,16],"lhs_strides":[1,30,60],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":15,"lhs_layout":"compact","lhs_shape":[31,2,16],"lhs_strides":[1,31,62],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":16,"lhs_layout":"compact","lhs_shape":[32,2,16],"lhs_strides":[1,32,64],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":17,"lhs_layout":"compact","lhs_shape":[33,2,16],"lhs_strides":[1,33,66],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":18,"lhs_layout":"compact","lhs_shape":[34,2,16],"lhs_strides":[1,34,68],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":19,"lhs_layout":"compact","lhs_shape":[35,2,16],"lhs_strides":[1,35,70],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":20,"lhs_layout":"compact","lhs_shape":[36,2,16],"lhs_strides":[1,36,72],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":21,"lhs_layout":"compact","lhs_shape":[37,2,16],"lhs_strides":[1,37,74],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":22,"lhs_layout":"compact","lhs_shape":[38,2,16],"lhs_strides":[1,38,76],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":23,"lhs_layout":"compact","lhs_shape":[39,2,16],"lhs_strides":[1,39,78],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":24,"lhs_layout":"compact","lhs_shape":[40,2,16],"lhs_strides":[1,40,80],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":25,"lhs_layout":"compact","lhs_shape":[41,2,16],"lhs_strides":[1,41,82],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":26,"lhs_layout":"compact","lhs_shape":[42,2,16],"lhs_strides":[1,42,84],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":27,"lhs_layout":"compact","lhs_shape":[43,2,16],"lhs_strides":[1,43,86],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":28,"lhs_layout":"compact","lhs_shape":[44,2,16],"lhs_strides":[1,44,88],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":29,"lhs_layout":"compact","lhs_shape":[45,2,16],"lhs_strides":[1,45,90],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":30,"lhs_layout":"compact","lhs_shape":[46,2,16],"lhs_strides":[1,46,92],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":31,"lhs_layout":"compact","lhs_shape":[47,2,16],"lhs_strides":[1,47,94],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":32,"lhs_layout":"compact","lhs_shape":[48,2,16],"lhs_strides":[1,48,96],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":33,"lhs_layout":"compact","lhs_shape":[49,2,16],"lhs_strides":[1,49,98],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":34,"lhs_layout":"compact","lhs_shape":[50,2,16],"lhs_strides":[1,50,100],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":35,"lhs_layout":"compact","lhs_shape":[51,2,16],"lhs_strides":[1,51,102],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":36,"lhs_layout":"compact","lhs_shape":[52,2,16],"lhs_strides":[1,52,104],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":37,"lhs_layout":"compact","lhs_shape":[53,2,16],"lhs_strides":[1,53,106],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":38,"lhs_layout":"compact","lhs_shape":[54,2,16],"lhs_strides":[1,54,108],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":39,"lhs_layout":"compact","lhs_shape":[55,2,16],"lhs_strides":[1,55,110],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":40,"lhs_layout":"compact","lhs_shape":[56,2,16],"lhs_strides":[1,56,112],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":41,"lhs_layout":"compact","lhs_shape":[57,2,16],"lhs_strides":[1,57,114],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":42,"lhs_layout":"compact","lhs_shape":[58,2,16],"lhs_strides":[1,58,116],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":43,"lhs_layout":"compact","lhs_shape":[59,2,16],"lhs_strides":[1,59,118],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":44,"lhs_layout":"compact","lhs_shape":[60,2,16],"lhs_strides":[1,60,120],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":45,"lhs_layout":"compact","lhs_shape":[61,2,16],"lhs_strides":[1,61,122],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":46,"lhs_layout":"compact","lhs_shape":[62,2,16],"lhs_strides":[1,62,124],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":47,"lhs_layout":"compact","lhs_shape":[63,2,16],"lhs_strides":[1,63,126],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":48,"lhs_layout":"compact","lhs_shape":[64,2,16],"lhs_strides":[1,64,128],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":49,"lhs_layout":"compact","lhs_shape":[65,2,16],"lhs_strides":[1,65,130],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":50,"lhs_layout":"compact","lhs_shape":[66,2,16],"lhs_strides":[1,66,132],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":51,"lhs_layout":"compact","lhs_shape":[67,2,16],"lhs_strides":[1,67,134],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":52,"lhs_layout":"compact","lhs_shape":[68,2,16],"lhs_strides":[1,68,136],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":53,"lhs_layout":"compact","lhs_shape":[69,2,16],"lhs_strides":[1,69,138],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":54,"lhs_layout":"compact","lhs_shape":[70,2,16],"lhs_strides":[1,70,140],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":55,"lhs_layout":"compact","lhs_shape":[71,2,16],"lhs_strides":[1,71,142],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":56,"lhs_layout":"compact","lhs_shape":[72,2,16],"lhs_strides":[1,72,144],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":57,"lhs_layout":"compact","lhs_shape":[73,2,16],"lhs_strides":[1,73,146],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":58,"lhs_layout":"compact","lhs_shape":[74,2,16],"lhs_strides":[1,74,148],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":59,"lhs_layout":"compact","lhs_shape":[75,2,16],"lhs_strides":[1,75,150],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":60,"lhs_layout":"compact","lhs_shape":[76,2,16],"lhs_strides":[1,76,152],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":61,"lhs_layout":"compact","lhs_shape":[77,2,16],"lhs_strides":[1,77,154],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":62,"lhs_layout":"compact","lhs_shape":[78,2,16],"lhs_strides":[1,78,156],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":63,"lhs_layout":"compact","lhs_shape":[79,2,16],"lhs_strides":[1,79,158],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":64,"lhs_layout":"compact","lhs_shape":[80,2,16],"lhs_strides":[1,80,160],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":65,"lhs_layout":"compact","lhs_shape":[81,2,16],"lhs_strides":[1,81,162],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":66,"lhs_layout":"compact","lhs_shape":[82,2,16],"lhs_strides":[1,82,164],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":67,"lhs_layout":"compact","lhs_shape":[83,2,16],"lhs_strides":[1,83,166],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":68,"lhs_layout":"compact","lhs_shape":[84,2,16],"lhs_strides":[1,84,168],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":69,"lhs_layout":"compact","lhs_shape":[85,2,16],"lhs_strides":[1,85,170],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":70,"lhs_layout":"compact","lhs_shape":[86,2,16],"lhs_strides":[1,86,172],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":71,"lhs_layout":"compact","lhs_shape":[87,2,16],"lhs_strides":[1,87,174],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]}]
```


### shape_stream_f64_n16_equal-shape-strides_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":0,"lhs_layout":"compact","lhs_shape":[16,2,16],"lhs_strides":[1,16,32],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]},{"key":1,"lhs_layout":"transposed","lhs_shape":[16,2,16],"lhs_strides":[32,16,1],"rhs_shape":[16,2,3],"rhs_strides":[1,16,32]}]
```


### shape_stream_f64_n64_fixed-warm_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]}]
```


### shape_stream_f64_n64_mixed-warm_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]}]
```


### shape_stream_f64_n64_fresh_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":3,"lhs_layout":"compact","lhs_shape":[67,2,64],"lhs_strides":[1,67,134],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":4,"lhs_layout":"compact","lhs_shape":[68,2,64],"lhs_strides":[1,68,136],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":5,"lhs_layout":"compact","lhs_shape":[69,2,64],"lhs_strides":[1,69,138],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":6,"lhs_layout":"compact","lhs_shape":[70,2,64],"lhs_strides":[1,70,140],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":7,"lhs_layout":"compact","lhs_shape":[71,2,64],"lhs_strides":[1,71,142],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]}]
```


### shape_stream_f64_n64_capacity-thrash_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"compact","lhs_shape":[65,2,64],"lhs_strides":[1,65,130],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":2,"lhs_layout":"compact","lhs_shape":[66,2,64],"lhs_strides":[1,66,132],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":3,"lhs_layout":"compact","lhs_shape":[67,2,64],"lhs_strides":[1,67,134],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":4,"lhs_layout":"compact","lhs_shape":[68,2,64],"lhs_strides":[1,68,136],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":5,"lhs_layout":"compact","lhs_shape":[69,2,64],"lhs_strides":[1,69,138],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":6,"lhs_layout":"compact","lhs_shape":[70,2,64],"lhs_strides":[1,70,140],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":7,"lhs_layout":"compact","lhs_shape":[71,2,64],"lhs_strides":[1,71,142],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":8,"lhs_layout":"compact","lhs_shape":[72,2,64],"lhs_strides":[1,72,144],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":9,"lhs_layout":"compact","lhs_shape":[73,2,64],"lhs_strides":[1,73,146],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":10,"lhs_layout":"compact","lhs_shape":[74,2,64],"lhs_strides":[1,74,148],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":11,"lhs_layout":"compact","lhs_shape":[75,2,64],"lhs_strides":[1,75,150],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":12,"lhs_layout":"compact","lhs_shape":[76,2,64],"lhs_strides":[1,76,152],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":13,"lhs_layout":"compact","lhs_shape":[77,2,64],"lhs_strides":[1,77,154],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":14,"lhs_layout":"compact","lhs_shape":[78,2,64],"lhs_strides":[1,78,156],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":15,"lhs_layout":"compact","lhs_shape":[79,2,64],"lhs_strides":[1,79,158],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":16,"lhs_layout":"compact","lhs_shape":[80,2,64],"lhs_strides":[1,80,160],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":17,"lhs_layout":"compact","lhs_shape":[81,2,64],"lhs_strides":[1,81,162],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":18,"lhs_layout":"compact","lhs_shape":[82,2,64],"lhs_strides":[1,82,164],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":19,"lhs_layout":"compact","lhs_shape":[83,2,64],"lhs_strides":[1,83,166],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":20,"lhs_layout":"compact","lhs_shape":[84,2,64],"lhs_strides":[1,84,168],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":21,"lhs_layout":"compact","lhs_shape":[85,2,64],"lhs_strides":[1,85,170],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":22,"lhs_layout":"compact","lhs_shape":[86,2,64],"lhs_strides":[1,86,172],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":23,"lhs_layout":"compact","lhs_shape":[87,2,64],"lhs_strides":[1,87,174],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":24,"lhs_layout":"compact","lhs_shape":[88,2,64],"lhs_strides":[1,88,176],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":25,"lhs_layout":"compact","lhs_shape":[89,2,64],"lhs_strides":[1,89,178],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":26,"lhs_layout":"compact","lhs_shape":[90,2,64],"lhs_strides":[1,90,180],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":27,"lhs_layout":"compact","lhs_shape":[91,2,64],"lhs_strides":[1,91,182],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":28,"lhs_layout":"compact","lhs_shape":[92,2,64],"lhs_strides":[1,92,184],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":29,"lhs_layout":"compact","lhs_shape":[93,2,64],"lhs_strides":[1,93,186],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":30,"lhs_layout":"compact","lhs_shape":[94,2,64],"lhs_strides":[1,94,188],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":31,"lhs_layout":"compact","lhs_shape":[95,2,64],"lhs_strides":[1,95,190],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":32,"lhs_layout":"compact","lhs_shape":[96,2,64],"lhs_strides":[1,96,192],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":33,"lhs_layout":"compact","lhs_shape":[97,2,64],"lhs_strides":[1,97,194],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":34,"lhs_layout":"compact","lhs_shape":[98,2,64],"lhs_strides":[1,98,196],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":35,"lhs_layout":"compact","lhs_shape":[99,2,64],"lhs_strides":[1,99,198],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":36,"lhs_layout":"compact","lhs_shape":[100,2,64],"lhs_strides":[1,100,200],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":37,"lhs_layout":"compact","lhs_shape":[101,2,64],"lhs_strides":[1,101,202],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":38,"lhs_layout":"compact","lhs_shape":[102,2,64],"lhs_strides":[1,102,204],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":39,"lhs_layout":"compact","lhs_shape":[103,2,64],"lhs_strides":[1,103,206],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":40,"lhs_layout":"compact","lhs_shape":[104,2,64],"lhs_strides":[1,104,208],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":41,"lhs_layout":"compact","lhs_shape":[105,2,64],"lhs_strides":[1,105,210],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":42,"lhs_layout":"compact","lhs_shape":[106,2,64],"lhs_strides":[1,106,212],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":43,"lhs_layout":"compact","lhs_shape":[107,2,64],"lhs_strides":[1,107,214],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":44,"lhs_layout":"compact","lhs_shape":[108,2,64],"lhs_strides":[1,108,216],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":45,"lhs_layout":"compact","lhs_shape":[109,2,64],"lhs_strides":[1,109,218],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":46,"lhs_layout":"compact","lhs_shape":[110,2,64],"lhs_strides":[1,110,220],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":47,"lhs_layout":"compact","lhs_shape":[111,2,64],"lhs_strides":[1,111,222],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":48,"lhs_layout":"compact","lhs_shape":[112,2,64],"lhs_strides":[1,112,224],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":49,"lhs_layout":"compact","lhs_shape":[113,2,64],"lhs_strides":[1,113,226],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":50,"lhs_layout":"compact","lhs_shape":[114,2,64],"lhs_strides":[1,114,228],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":51,"lhs_layout":"compact","lhs_shape":[115,2,64],"lhs_strides":[1,115,230],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":52,"lhs_layout":"compact","lhs_shape":[116,2,64],"lhs_strides":[1,116,232],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":53,"lhs_layout":"compact","lhs_shape":[117,2,64],"lhs_strides":[1,117,234],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":54,"lhs_layout":"compact","lhs_shape":[118,2,64],"lhs_strides":[1,118,236],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":55,"lhs_layout":"compact","lhs_shape":[119,2,64],"lhs_strides":[1,119,238],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":56,"lhs_layout":"compact","lhs_shape":[120,2,64],"lhs_strides":[1,120,240],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":57,"lhs_layout":"compact","lhs_shape":[121,2,64],"lhs_strides":[1,121,242],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":58,"lhs_layout":"compact","lhs_shape":[122,2,64],"lhs_strides":[1,122,244],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":59,"lhs_layout":"compact","lhs_shape":[123,2,64],"lhs_strides":[1,123,246],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":60,"lhs_layout":"compact","lhs_shape":[124,2,64],"lhs_strides":[1,124,248],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":61,"lhs_layout":"compact","lhs_shape":[125,2,64],"lhs_strides":[1,125,250],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":62,"lhs_layout":"compact","lhs_shape":[126,2,64],"lhs_strides":[1,126,252],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":63,"lhs_layout":"compact","lhs_shape":[127,2,64],"lhs_strides":[1,127,254],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":64,"lhs_layout":"compact","lhs_shape":[128,2,64],"lhs_strides":[1,128,256],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":65,"lhs_layout":"compact","lhs_shape":[129,2,64],"lhs_strides":[1,129,258],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":66,"lhs_layout":"compact","lhs_shape":[130,2,64],"lhs_strides":[1,130,260],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":67,"lhs_layout":"compact","lhs_shape":[131,2,64],"lhs_strides":[1,131,262],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":68,"lhs_layout":"compact","lhs_shape":[132,2,64],"lhs_strides":[1,132,264],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":69,"lhs_layout":"compact","lhs_shape":[133,2,64],"lhs_strides":[1,133,266],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":70,"lhs_layout":"compact","lhs_shape":[134,2,64],"lhs_strides":[1,134,268],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":71,"lhs_layout":"compact","lhs_shape":[135,2,64],"lhs_strides":[1,135,270],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]}]
```


### shape_stream_f64_n64_equal-shape-strides_latency sequence

```json
[{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":0,"lhs_layout":"compact","lhs_shape":[64,2,64],"lhs_strides":[1,64,128],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]},{"key":1,"lhs_layout":"transposed","lhs_shape":[64,2,64],"lhs_strides":[128,64,1],"rhs_shape":[64,2,3],"rhs_strides":[1,64,128]}]
```


## Counter diagnostics (no timing)

### shape_stream_f64_n16_fixed-warm_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16782080 | 33554432 | 0 | 36 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 16782848 | 33554432 | 0 | 37 | 0 | 0 | 0 |
| enqueue | 1 | 0 | 16783616 | 33554432 | 0 | 38 | 0 | 0 | 0 |
| enqueue | 2 | 0 | 16784384 | 33554432 | 0 | 39 | 0 | 0 | 0 |
| enqueue | 3 | 0 | 16785152 | 33554432 | 0 | 40 | 0 | 0 | 0 |
| enqueue | 4 | 0 | 16785920 | 33554432 | 0 | 41 | 0 | 0 | 0 |
| enqueue | 5 | 0 | 16786688 | 33554432 | 0 | 42 | 0 | 0 | 0 |
| enqueue | 6 | 0 | 16787456 | 33554432 | 0 | 43 | 0 | 0 | 0 |
| enqueue | 7 | 0 | 16788224 | 33554432 | 0 | 44 | 0 | 0 | 0 |
| enqueue | 8 | 0 | 16788992 | 33554432 | 0 | 45 | 0 | 0 | 0 |
| enqueue | 9 | 0 | 16789760 | 33554432 | 0 | 46 | 0 | 0 | 0 |
| enqueue | 10 | 0 | 16790528 | 33554432 | 0 | 47 | 0 | 0 | 0 |
| enqueue | 11 | 0 | 16791296 | 33554432 | 0 | 48 | 0 | 0 | 0 |
| enqueue | 12 | 0 | 16792064 | 33554432 | 0 | 49 | 0 | 0 | 0 |
| enqueue | 13 | 0 | 16792832 | 33554432 | 0 | 50 | 0 | 0 | 0 |
| enqueue | 14 | 0 | 16793600 | 33554432 | 0 | 51 | 0 | 0 | 0 |
| enqueue | 15 | 0 | 16794368 | 33554432 | 0 | 52 | 0 | 0 | 0 |
| enqueue | 16 | 0 | 16795136 | 33554432 | 0 | 53 | 0 | 0 | 0 |
| enqueue | 17 | 0 | 16795904 | 33554432 | 0 | 54 | 0 | 0 | 0 |
| enqueue | 18 | 0 | 16796672 | 33554432 | 0 | 55 | 0 | 0 | 0 |
| enqueue | 19 | 0 | 16797440 | 33554432 | 0 | 56 | 0 | 0 | 0 |
| enqueue | 20 | 0 | 16798208 | 33554432 | 0 | 57 | 0 | 0 | 0 |
| enqueue | 21 | 0 | 16798976 | 33554432 | 0 | 58 | 0 | 0 | 0 |
| enqueue | 22 | 0 | 16799744 | 33554432 | 0 | 59 | 0 | 0 | 0 |
| enqueue | 23 | 0 | 16800512 | 33554432 | 0 | 60 | 0 | 0 | 0 |
| enqueue | 24 | 0 | 16801280 | 33554432 | 0 | 61 | 0 | 0 | 0 |
| enqueue | 25 | 0 | 16802048 | 33554432 | 0 | 62 | 0 | 0 | 0 |
| enqueue | 26 | 0 | 16802816 | 33554432 | 0 | 63 | 0 | 0 | 0 |
| enqueue | 27 | 0 | 16803584 | 33554432 | 0 | 64 | 0 | 0 | 0 |
| enqueue | 28 | 0 | 16804352 | 33554432 | 0 | 65 | 0 | 0 | 0 |
| enqueue | 29 | 0 | 16805120 | 33554432 | 0 | 66 | 0 | 0 | 0 |
| enqueue | 30 | 0 | 16805888 | 33554432 | 0 | 67 | 0 | 0 | 0 |
| enqueue | 31 | 0 | 16806656 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| completion | — | — | 16806656 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| drop | — | — | 16782080 | 33554432 | 0 | 68 | 0 | 0 | 0 |

### shape_stream_f64_n16_mixed-warm_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16792576 | 33554432 | 0 | 42 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 16793344 | 33554432 | 0 | 43 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 16794160 | 33554432 | 0 | 44 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 16795024 | 33554432 | 0 | 45 | 0 | 0 | 0 |
| enqueue | 3 | 0 | 16795792 | 33554432 | 0 | 46 | 0 | 0 | 0 |
| enqueue | 4 | 2 | 16796656 | 33554432 | 0 | 47 | 0 | 0 | 0 |
| enqueue | 5 | 1 | 16797472 | 33554432 | 0 | 48 | 0 | 0 | 0 |
| enqueue | 6 | 0 | 16798240 | 33554432 | 0 | 49 | 0 | 0 | 0 |
| enqueue | 7 | 1 | 16799056 | 33554432 | 0 | 50 | 0 | 0 | 0 |
| enqueue | 8 | 2 | 16799920 | 33554432 | 0 | 51 | 0 | 0 | 0 |
| enqueue | 9 | 0 | 16800688 | 33554432 | 0 | 52 | 0 | 0 | 0 |
| enqueue | 10 | 2 | 16801552 | 33554432 | 0 | 53 | 0 | 0 | 0 |
| enqueue | 11 | 1 | 16802368 | 33554432 | 0 | 54 | 0 | 0 | 0 |
| enqueue | 12 | 0 | 16803136 | 33554432 | 0 | 55 | 0 | 0 | 0 |
| enqueue | 13 | 1 | 16803952 | 33554432 | 0 | 56 | 0 | 0 | 0 |
| enqueue | 14 | 2 | 16804816 | 33554432 | 0 | 57 | 0 | 0 | 0 |
| enqueue | 15 | 0 | 16805584 | 33554432 | 0 | 58 | 0 | 0 | 0 |
| enqueue | 16 | 2 | 16806448 | 33554432 | 0 | 59 | 0 | 0 | 0 |
| enqueue | 17 | 1 | 16807264 | 33554432 | 0 | 60 | 0 | 0 | 0 |
| enqueue | 18 | 0 | 16808032 | 33554432 | 0 | 61 | 0 | 0 | 0 |
| enqueue | 19 | 1 | 16808848 | 33554432 | 0 | 62 | 0 | 0 | 0 |
| enqueue | 20 | 2 | 16809712 | 33554432 | 0 | 63 | 0 | 0 | 0 |
| enqueue | 21 | 0 | 16810480 | 33554432 | 0 | 64 | 0 | 0 | 0 |
| enqueue | 22 | 2 | 16811344 | 33554432 | 0 | 65 | 0 | 0 | 0 |
| enqueue | 23 | 1 | 16812160 | 33554432 | 0 | 66 | 0 | 0 | 0 |
| enqueue | 24 | 0 | 16812928 | 33554432 | 0 | 67 | 0 | 0 | 0 |
| enqueue | 25 | 1 | 16813744 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| enqueue | 26 | 2 | 16814608 | 33554432 | 0 | 69 | 0 | 0 | 0 |
| enqueue | 27 | 0 | 16815376 | 33554432 | 0 | 70 | 0 | 0 | 0 |
| enqueue | 28 | 2 | 16816240 | 33554432 | 0 | 71 | 0 | 0 | 0 |
| enqueue | 29 | 1 | 16817056 | 33554432 | 0 | 72 | 0 | 0 | 0 |
| enqueue | 30 | 0 | 16817824 | 33554432 | 0 | 73 | 0 | 0 | 0 |
| enqueue | 31 | 1 | 16818640 | 33554432 | 0 | 74 | 0 | 0 | 0 |
| completion | — | — | 16818640 | 33554432 | 0 | 74 | 0 | 0 | 0 |
| drop | — | — | 16792576 | 33554432 | 0 | 74 | 0 | 0 | 0 |

### shape_stream_f64_n16_fresh_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16823296 | 33554432 | 0 | 25 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 16824064 | 33554432 | 0 | 26 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 16824880 | 33554432 | 0 | 27 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 16825744 | 33554432 | 0 | 28 | 0 | 0 | 0 |
| enqueue | 3 | 3 | 16826656 | 33554432 | 0 | 29 | 0 | 0 | 0 |
| enqueue | 4 | 4 | 16827616 | 33554432 | 0 | 30 | 0 | 0 | 0 |
| enqueue | 5 | 5 | 16828624 | 33554432 | 0 | 31 | 0 | 0 | 0 |
| enqueue | 6 | 6 | 16829680 | 33554432 | 0 | 32 | 0 | 0 | 0 |
| enqueue | 7 | 7 | 16830784 | 33554432 | 0 | 33 | 0 | 0 | 0 |
| completion | — | — | 16830784 | 33554432 | 0 | 33 | 0 | 0 | 0 |
| drop | — | — | 16823296 | 33554432 | 0 | 33 | 0 | 0 | 0 |

### shape_stream_f64_n16_capacity-thrash_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 17781760 | 33554432 | 0 | 218 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 17782528 | 33554432 | 0 | 219 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 17783344 | 33554432 | 0 | 220 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 17784208 | 33554432 | 0 | 221 | 0 | 0 | 0 |
| enqueue | 3 | 3 | 17785120 | 33554432 | 0 | 222 | 0 | 0 | 0 |
| enqueue | 4 | 4 | 17786080 | 33554432 | 0 | 223 | 0 | 0 | 0 |
| enqueue | 5 | 5 | 17787088 | 33554432 | 0 | 224 | 0 | 0 | 0 |
| enqueue | 6 | 6 | 17788144 | 33554432 | 0 | 225 | 0 | 0 | 0 |
| enqueue | 7 | 7 | 17789248 | 33554432 | 0 | 226 | 0 | 0 | 0 |
| enqueue | 8 | 8 | 17790400 | 33554432 | 0 | 227 | 0 | 0 | 0 |
| enqueue | 9 | 9 | 17791600 | 33554432 | 0 | 228 | 0 | 0 | 0 |
| enqueue | 10 | 10 | 17792848 | 33554432 | 0 | 229 | 0 | 0 | 0 |
| enqueue | 11 | 11 | 17794144 | 33554432 | 0 | 230 | 0 | 0 | 0 |
| enqueue | 12 | 12 | 17795488 | 33554432 | 0 | 231 | 0 | 0 | 0 |
| enqueue | 13 | 13 | 17796880 | 33554432 | 0 | 232 | 0 | 0 | 0 |
| enqueue | 14 | 14 | 17798320 | 33554432 | 0 | 233 | 0 | 0 | 0 |
| enqueue | 15 | 15 | 17799808 | 33554432 | 0 | 234 | 0 | 0 | 0 |
| enqueue | 16 | 16 | 17801344 | 33554432 | 0 | 235 | 0 | 0 | 0 |
| enqueue | 17 | 17 | 17802928 | 33554432 | 0 | 236 | 0 | 0 | 0 |
| enqueue | 18 | 18 | 17804560 | 33554432 | 0 | 237 | 0 | 0 | 0 |
| enqueue | 19 | 19 | 17806240 | 33554432 | 0 | 238 | 0 | 0 | 0 |
| enqueue | 20 | 20 | 17807968 | 33554432 | 0 | 239 | 0 | 0 | 0 |
| enqueue | 21 | 21 | 17809744 | 33554432 | 0 | 240 | 0 | 0 | 0 |
| enqueue | 22 | 22 | 17811568 | 33554432 | 0 | 241 | 0 | 0 | 0 |
| enqueue | 23 | 23 | 17813440 | 33554432 | 0 | 242 | 0 | 0 | 0 |
| enqueue | 24 | 24 | 17815360 | 33554432 | 0 | 243 | 0 | 0 | 0 |
| enqueue | 25 | 25 | 17817328 | 33554432 | 0 | 244 | 0 | 0 | 0 |
| enqueue | 26 | 26 | 17819344 | 33554432 | 0 | 245 | 0 | 0 | 0 |
| enqueue | 27 | 27 | 17821408 | 33554432 | 0 | 246 | 0 | 0 | 0 |
| enqueue | 28 | 28 | 17823520 | 33554432 | 0 | 247 | 0 | 0 | 0 |
| enqueue | 29 | 29 | 17825680 | 33554432 | 0 | 248 | 0 | 0 | 0 |
| enqueue | 30 | 30 | 17827888 | 33554432 | 0 | 249 | 0 | 0 | 0 |
| enqueue | 31 | 31 | 17830144 | 33554432 | 0 | 250 | 0 | 0 | 0 |
| enqueue | 32 | 32 | 17832448 | 33554432 | 0 | 251 | 0 | 0 | 0 |
| enqueue | 33 | 33 | 17834800 | 33554432 | 0 | 252 | 0 | 0 | 0 |
| enqueue | 34 | 34 | 17837200 | 33554432 | 0 | 253 | 0 | 0 | 0 |
| enqueue | 35 | 35 | 17839648 | 33554432 | 0 | 254 | 0 | 0 | 0 |
| enqueue | 36 | 36 | 17842144 | 33554432 | 0 | 255 | 0 | 0 | 0 |
| enqueue | 37 | 37 | 17844688 | 33554432 | 0 | 256 | 0 | 0 | 0 |
| enqueue | 38 | 38 | 17847280 | 33554432 | 0 | 257 | 0 | 0 | 0 |
| enqueue | 39 | 39 | 17849920 | 33554432 | 0 | 258 | 0 | 0 | 0 |
| enqueue | 40 | 40 | 17852608 | 33554432 | 0 | 259 | 0 | 0 | 0 |
| enqueue | 41 | 41 | 17855344 | 33554432 | 0 | 260 | 0 | 0 | 0 |
| enqueue | 42 | 42 | 17858128 | 33554432 | 0 | 261 | 0 | 0 | 0 |
| enqueue | 43 | 43 | 17860960 | 33554432 | 0 | 262 | 0 | 0 | 0 |
| enqueue | 44 | 44 | 17863840 | 33554432 | 0 | 263 | 0 | 0 | 0 |
| enqueue | 45 | 45 | 17866768 | 33554432 | 0 | 264 | 0 | 0 | 0 |
| enqueue | 46 | 46 | 17869744 | 33554432 | 0 | 265 | 0 | 0 | 0 |
| enqueue | 47 | 47 | 17872768 | 33554432 | 0 | 266 | 0 | 0 | 0 |
| enqueue | 48 | 48 | 17875840 | 33554432 | 0 | 267 | 0 | 0 | 0 |
| enqueue | 49 | 49 | 17878960 | 33554432 | 0 | 268 | 0 | 0 | 0 |
| enqueue | 50 | 50 | 17882128 | 33554432 | 0 | 269 | 0 | 0 | 0 |
| enqueue | 51 | 51 | 17885344 | 33554432 | 0 | 270 | 0 | 0 | 0 |
| enqueue | 52 | 52 | 17888608 | 33554432 | 0 | 271 | 0 | 0 | 0 |
| enqueue | 53 | 53 | 17891920 | 33554432 | 0 | 272 | 0 | 0 | 0 |
| enqueue | 54 | 54 | 17895280 | 33554432 | 0 | 273 | 0 | 0 | 0 |
| enqueue | 55 | 55 | 17898688 | 33554432 | 0 | 274 | 0 | 0 | 0 |
| enqueue | 56 | 56 | 17902144 | 33554432 | 0 | 275 | 0 | 0 | 0 |
| enqueue | 57 | 57 | 17905648 | 33554432 | 0 | 276 | 0 | 0 | 0 |
| enqueue | 58 | 58 | 17909200 | 33554432 | 0 | 277 | 0 | 0 | 0 |
| enqueue | 59 | 59 | 17912800 | 33554432 | 0 | 278 | 0 | 0 | 0 |
| enqueue | 60 | 60 | 17916448 | 33554432 | 0 | 279 | 0 | 0 | 0 |
| enqueue | 61 | 61 | 17920144 | 33554432 | 0 | 280 | 0 | 0 | 0 |
| enqueue | 62 | 62 | 17923888 | 33554432 | 0 | 281 | 0 | 0 | 0 |
| enqueue | 63 | 63 | 17927680 | 33554432 | 0 | 282 | 0 | 0 | 0 |
| enqueue | 64 | 64 | 17931520 | 33554432 | 0 | 283 | 0 | 0 | 0 |
| enqueue | 65 | 65 | 17935408 | 33554432 | 0 | 284 | 0 | 0 | 0 |
| enqueue | 66 | 66 | 17939344 | 33554432 | 0 | 285 | 0 | 0 | 0 |
| enqueue | 67 | 67 | 17943328 | 33554432 | 0 | 286 | 0 | 0 | 0 |
| enqueue | 68 | 68 | 17947360 | 33554432 | 0 | 287 | 0 | 0 | 0 |
| enqueue | 69 | 69 | 17951440 | 33554432 | 0 | 288 | 0 | 0 | 0 |
| enqueue | 70 | 70 | 17955568 | 33554432 | 0 | 289 | 0 | 0 | 0 |
| enqueue | 71 | 71 | 17959744 | 33554432 | 0 | 290 | 0 | 0 | 0 |
| enqueue | 72 | 0 | 17960512 | 33554432 | 0 | 291 | 0 | 0 | 0 |
| completion | — | — | 17960512 | 33554432 | 0 | 291 | 0 | 0 | 0 |
| drop | — | — | 17781760 | 33554432 | 0 | 291 | 0 | 0 | 0 |

### shape_stream_f64_n16_equal-shape-strides_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16786944 | 33554432 | 0 | 56 | 17 | 1 | 1 |
| enqueue | 0 | 0 | 16787712 | 33554432 | 0 | 57 | 17 | 1 | 1 |
| enqueue | 1 | 1 | 16788480 | 33554432 | 0 | 59 | 18 | 1 | 1 |
| enqueue | 2 | 0 | 16789248 | 33554432 | 0 | 60 | 18 | 1 | 1 |
| enqueue | 3 | 1 | 16790016 | 33554432 | 0 | 62 | 19 | 1 | 1 |
| enqueue | 4 | 0 | 16790784 | 33554432 | 0 | 63 | 19 | 1 | 1 |
| enqueue | 5 | 1 | 16791552 | 33554432 | 0 | 65 | 20 | 1 | 1 |
| enqueue | 6 | 0 | 16792320 | 33554432 | 0 | 66 | 20 | 1 | 1 |
| enqueue | 7 | 1 | 16793088 | 33554432 | 0 | 68 | 21 | 1 | 1 |
| enqueue | 8 | 0 | 16793856 | 33554432 | 0 | 69 | 21 | 1 | 1 |
| enqueue | 9 | 1 | 16794624 | 33554432 | 0 | 71 | 22 | 1 | 1 |
| enqueue | 10 | 0 | 16795392 | 33554432 | 0 | 72 | 22 | 1 | 1 |
| enqueue | 11 | 1 | 16796160 | 33554432 | 0 | 74 | 23 | 1 | 1 |
| enqueue | 12 | 0 | 16796928 | 33554432 | 0 | 75 | 23 | 1 | 1 |
| enqueue | 13 | 1 | 16797696 | 33554432 | 0 | 77 | 24 | 1 | 1 |
| enqueue | 14 | 0 | 16798464 | 33554432 | 0 | 78 | 24 | 1 | 1 |
| enqueue | 15 | 1 | 16799232 | 33554432 | 0 | 80 | 25 | 1 | 1 |
| enqueue | 16 | 0 | 16800000 | 33554432 | 0 | 81 | 25 | 1 | 1 |
| enqueue | 17 | 1 | 16800768 | 33554432 | 0 | 83 | 26 | 1 | 1 |
| enqueue | 18 | 0 | 16801536 | 33554432 | 0 | 84 | 26 | 1 | 1 |
| enqueue | 19 | 1 | 16802304 | 33554432 | 0 | 86 | 27 | 1 | 1 |
| enqueue | 20 | 0 | 16803072 | 33554432 | 0 | 87 | 27 | 1 | 1 |
| enqueue | 21 | 1 | 16803840 | 33554432 | 0 | 89 | 28 | 1 | 1 |
| enqueue | 22 | 0 | 16804608 | 33554432 | 0 | 90 | 28 | 1 | 1 |
| enqueue | 23 | 1 | 16805376 | 33554432 | 0 | 92 | 29 | 1 | 1 |
| enqueue | 24 | 0 | 16806144 | 33554432 | 0 | 93 | 29 | 1 | 1 |
| enqueue | 25 | 1 | 16806912 | 33554432 | 0 | 95 | 30 | 1 | 1 |
| enqueue | 26 | 0 | 16807680 | 33554432 | 0 | 96 | 30 | 1 | 1 |
| enqueue | 27 | 1 | 16808448 | 33554432 | 0 | 98 | 31 | 1 | 1 |
| enqueue | 28 | 0 | 16809216 | 33554432 | 0 | 99 | 31 | 1 | 1 |
| enqueue | 29 | 1 | 16809984 | 33554432 | 0 | 101 | 32 | 1 | 1 |
| enqueue | 30 | 0 | 16810752 | 33554432 | 0 | 102 | 32 | 1 | 1 |
| enqueue | 31 | 1 | 16811520 | 33554432 | 0 | 104 | 33 | 1 | 1 |
| completion | — | — | 16811520 | 33554432 | 0 | 104 | 33 | 1 | 1 |
| drop | — | — | 16786944 | 33554432 | 0 | 104 | 33 | 1 | 1 |

### shape_stream_f64_n64_fixed-warm_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16845824 | 33554432 | 0 | 36 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 16848896 | 33554432 | 0 | 37 | 0 | 0 | 0 |
| enqueue | 1 | 0 | 16851968 | 33554432 | 0 | 38 | 0 | 0 | 0 |
| enqueue | 2 | 0 | 16855040 | 33554432 | 0 | 39 | 0 | 0 | 0 |
| enqueue | 3 | 0 | 16858112 | 33554432 | 0 | 40 | 0 | 0 | 0 |
| enqueue | 4 | 0 | 16861184 | 33554432 | 0 | 41 | 0 | 0 | 0 |
| enqueue | 5 | 0 | 16864256 | 33554432 | 0 | 42 | 0 | 0 | 0 |
| enqueue | 6 | 0 | 16867328 | 33554432 | 0 | 43 | 0 | 0 | 0 |
| enqueue | 7 | 0 | 16870400 | 33554432 | 0 | 44 | 0 | 0 | 0 |
| enqueue | 8 | 0 | 16873472 | 33554432 | 0 | 45 | 0 | 0 | 0 |
| enqueue | 9 | 0 | 16876544 | 33554432 | 0 | 46 | 0 | 0 | 0 |
| enqueue | 10 | 0 | 16879616 | 33554432 | 0 | 47 | 0 | 0 | 0 |
| enqueue | 11 | 0 | 16882688 | 33554432 | 0 | 48 | 0 | 0 | 0 |
| enqueue | 12 | 0 | 16885760 | 33554432 | 0 | 49 | 0 | 0 | 0 |
| enqueue | 13 | 0 | 16888832 | 33554432 | 0 | 50 | 0 | 0 | 0 |
| enqueue | 14 | 0 | 16891904 | 33554432 | 0 | 51 | 0 | 0 | 0 |
| enqueue | 15 | 0 | 16894976 | 33554432 | 0 | 52 | 0 | 0 | 0 |
| enqueue | 16 | 0 | 16898048 | 33554432 | 0 | 53 | 0 | 0 | 0 |
| enqueue | 17 | 0 | 16901120 | 33554432 | 0 | 54 | 0 | 0 | 0 |
| enqueue | 18 | 0 | 16904192 | 33554432 | 0 | 55 | 0 | 0 | 0 |
| enqueue | 19 | 0 | 16907264 | 33554432 | 0 | 56 | 0 | 0 | 0 |
| enqueue | 20 | 0 | 16910336 | 33554432 | 0 | 57 | 0 | 0 | 0 |
| enqueue | 21 | 0 | 16913408 | 33554432 | 0 | 58 | 0 | 0 | 0 |
| enqueue | 22 | 0 | 16916480 | 33554432 | 0 | 59 | 0 | 0 | 0 |
| enqueue | 23 | 0 | 16919552 | 33554432 | 0 | 60 | 0 | 0 | 0 |
| enqueue | 24 | 0 | 16922624 | 33554432 | 0 | 61 | 0 | 0 | 0 |
| enqueue | 25 | 0 | 16925696 | 33554432 | 0 | 62 | 0 | 0 | 0 |
| enqueue | 26 | 0 | 16928768 | 33554432 | 0 | 63 | 0 | 0 | 0 |
| enqueue | 27 | 0 | 16931840 | 33554432 | 0 | 64 | 0 | 0 | 0 |
| enqueue | 28 | 0 | 16934912 | 33554432 | 0 | 65 | 0 | 0 | 0 |
| enqueue | 29 | 0 | 16937984 | 33554432 | 0 | 66 | 0 | 0 | 0 |
| enqueue | 30 | 0 | 16941056 | 33554432 | 0 | 67 | 0 | 0 | 0 |
| enqueue | 31 | 0 | 16944128 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| completion | — | — | 16944128 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| drop | — | — | 16845824 | 33554432 | 0 | 68 | 0 | 0 | 0 |

### shape_stream_f64_n64_mixed-warm_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16986112 | 33554432 | 0 | 42 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 16989184 | 33554432 | 0 | 43 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 16992304 | 33554432 | 0 | 44 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 16995472 | 33554432 | 0 | 45 | 0 | 0 | 0 |
| enqueue | 3 | 0 | 16998544 | 33554432 | 0 | 46 | 0 | 0 | 0 |
| enqueue | 4 | 2 | 17001712 | 33554432 | 0 | 47 | 0 | 0 | 0 |
| enqueue | 5 | 1 | 17004832 | 33554432 | 0 | 48 | 0 | 0 | 0 |
| enqueue | 6 | 0 | 17007904 | 33554432 | 0 | 49 | 0 | 0 | 0 |
| enqueue | 7 | 1 | 17011024 | 33554432 | 0 | 50 | 0 | 0 | 0 |
| enqueue | 8 | 2 | 17014192 | 33554432 | 0 | 51 | 0 | 0 | 0 |
| enqueue | 9 | 0 | 17017264 | 33554432 | 0 | 52 | 0 | 0 | 0 |
| enqueue | 10 | 2 | 17020432 | 33554432 | 0 | 53 | 0 | 0 | 0 |
| enqueue | 11 | 1 | 17023552 | 33554432 | 0 | 54 | 0 | 0 | 0 |
| enqueue | 12 | 0 | 17026624 | 33554432 | 0 | 55 | 0 | 0 | 0 |
| enqueue | 13 | 1 | 17029744 | 33554432 | 0 | 56 | 0 | 0 | 0 |
| enqueue | 14 | 2 | 17032912 | 33554432 | 0 | 57 | 0 | 0 | 0 |
| enqueue | 15 | 0 | 17035984 | 33554432 | 0 | 58 | 0 | 0 | 0 |
| enqueue | 16 | 2 | 17039152 | 33554432 | 0 | 59 | 0 | 0 | 0 |
| enqueue | 17 | 1 | 17042272 | 33554432 | 0 | 60 | 0 | 0 | 0 |
| enqueue | 18 | 0 | 17045344 | 33554432 | 0 | 61 | 0 | 0 | 0 |
| enqueue | 19 | 1 | 17048464 | 33554432 | 0 | 62 | 0 | 0 | 0 |
| enqueue | 20 | 2 | 17051632 | 33554432 | 0 | 63 | 0 | 0 | 0 |
| enqueue | 21 | 0 | 17054704 | 33554432 | 0 | 64 | 0 | 0 | 0 |
| enqueue | 22 | 2 | 17057872 | 33554432 | 0 | 65 | 0 | 0 | 0 |
| enqueue | 23 | 1 | 17060992 | 33554432 | 0 | 66 | 0 | 0 | 0 |
| enqueue | 24 | 0 | 17064064 | 33554432 | 0 | 67 | 0 | 0 | 0 |
| enqueue | 25 | 1 | 17067184 | 33554432 | 0 | 68 | 0 | 0 | 0 |
| enqueue | 26 | 2 | 17070352 | 33554432 | 0 | 69 | 0 | 0 | 0 |
| enqueue | 27 | 0 | 17073424 | 33554432 | 0 | 70 | 0 | 0 | 0 |
| enqueue | 28 | 2 | 17076592 | 33554432 | 0 | 71 | 0 | 0 | 0 |
| enqueue | 29 | 1 | 17079712 | 33554432 | 0 | 72 | 0 | 0 | 0 |
| enqueue | 30 | 0 | 17082784 | 33554432 | 0 | 73 | 0 | 0 | 0 |
| enqueue | 31 | 1 | 17085904 | 33554432 | 0 | 74 | 0 | 0 | 0 |
| completion | — | — | 17085904 | 33554432 | 0 | 74 | 0 | 0 | 0 |
| drop | — | — | 16986112 | 33554432 | 0 | 74 | 0 | 0 | 0 |

### shape_stream_f64_n64_fresh_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 17354752 | 33554432 | 0 | 25 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 17357824 | 33554432 | 0 | 26 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 17360944 | 33554432 | 0 | 27 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 17364112 | 33554432 | 0 | 28 | 0 | 0 | 0 |
| enqueue | 3 | 3 | 17367328 | 33554432 | 0 | 29 | 0 | 0 | 0 |
| enqueue | 4 | 4 | 17370592 | 33554432 | 0 | 30 | 0 | 0 | 0 |
| enqueue | 5 | 5 | 17373904 | 33554432 | 0 | 31 | 0 | 0 | 0 |
| enqueue | 6 | 6 | 17377264 | 33554432 | 0 | 32 | 0 | 0 | 0 |
| enqueue | 7 | 7 | 17380672 | 33554432 | 0 | 33 | 0 | 0 | 0 |
| completion | — | — | 17380672 | 33554432 | 0 | 33 | 0 | 0 | 0 |
| drop | — | — | 17354752 | 33554432 | 0 | 33 | 0 | 0 | 0 |

### shape_stream_f64_n64_capacity-thrash_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 24334336 | 33554432 | 0 | 218 | 0 | 0 | 0 |
| enqueue | 0 | 0 | 24337408 | 33554432 | 0 | 219 | 0 | 0 | 0 |
| enqueue | 1 | 1 | 24340528 | 33554432 | 0 | 220 | 0 | 0 | 0 |
| enqueue | 2 | 2 | 24343696 | 33554432 | 0 | 221 | 0 | 0 | 0 |
| enqueue | 3 | 3 | 24346912 | 33554432 | 0 | 222 | 0 | 0 | 0 |
| enqueue | 4 | 4 | 24350176 | 33554432 | 0 | 223 | 0 | 0 | 0 |
| enqueue | 5 | 5 | 24353488 | 33554432 | 0 | 224 | 0 | 0 | 0 |
| enqueue | 6 | 6 | 24356848 | 33554432 | 0 | 225 | 0 | 0 | 0 |
| enqueue | 7 | 7 | 24360256 | 33554432 | 0 | 226 | 0 | 0 | 0 |
| enqueue | 8 | 8 | 24363712 | 33554432 | 0 | 227 | 0 | 0 | 0 |
| enqueue | 9 | 9 | 24367216 | 33554432 | 0 | 228 | 0 | 0 | 0 |
| enqueue | 10 | 10 | 24370768 | 33554432 | 0 | 229 | 0 | 0 | 0 |
| enqueue | 11 | 11 | 24374368 | 33554432 | 0 | 230 | 0 | 0 | 0 |
| enqueue | 12 | 12 | 24378016 | 33554432 | 0 | 231 | 0 | 0 | 0 |
| enqueue | 13 | 13 | 24381712 | 33554432 | 0 | 232 | 0 | 0 | 0 |
| enqueue | 14 | 14 | 24385456 | 33554432 | 0 | 233 | 0 | 0 | 0 |
| enqueue | 15 | 15 | 24389248 | 33554432 | 0 | 234 | 0 | 0 | 0 |
| enqueue | 16 | 16 | 24393088 | 33554432 | 0 | 235 | 0 | 0 | 0 |
| enqueue | 17 | 17 | 24396976 | 33554432 | 0 | 236 | 0 | 0 | 0 |
| enqueue | 18 | 18 | 24400912 | 33554432 | 0 | 237 | 0 | 0 | 0 |
| enqueue | 19 | 19 | 24404896 | 33554432 | 0 | 238 | 0 | 0 | 0 |
| enqueue | 20 | 20 | 24408928 | 33554432 | 0 | 239 | 0 | 0 | 0 |
| enqueue | 21 | 21 | 24413008 | 33554432 | 0 | 240 | 0 | 0 | 0 |
| enqueue | 22 | 22 | 24417136 | 33554432 | 0 | 241 | 0 | 0 | 0 |
| enqueue | 23 | 23 | 24421312 | 33554432 | 0 | 242 | 0 | 0 | 0 |
| enqueue | 24 | 24 | 24425536 | 33554432 | 0 | 243 | 0 | 0 | 0 |
| enqueue | 25 | 25 | 24429808 | 33554432 | 0 | 244 | 0 | 0 | 0 |
| enqueue | 26 | 26 | 24434128 | 33554432 | 0 | 245 | 0 | 0 | 0 |
| enqueue | 27 | 27 | 24438496 | 33554432 | 0 | 246 | 0 | 0 | 0 |
| enqueue | 28 | 28 | 24442912 | 33554432 | 0 | 247 | 0 | 0 | 0 |
| enqueue | 29 | 29 | 24447376 | 33554432 | 0 | 248 | 0 | 0 | 0 |
| enqueue | 30 | 30 | 24451888 | 33554432 | 0 | 249 | 0 | 0 | 0 |
| enqueue | 31 | 31 | 24456448 | 33554432 | 0 | 250 | 0 | 0 | 0 |
| enqueue | 32 | 32 | 24461056 | 33554432 | 0 | 251 | 0 | 0 | 0 |
| enqueue | 33 | 33 | 24465712 | 33554432 | 0 | 252 | 0 | 0 | 0 |
| enqueue | 34 | 34 | 24470416 | 33554432 | 0 | 253 | 0 | 0 | 0 |
| enqueue | 35 | 35 | 24475168 | 33554432 | 0 | 254 | 0 | 0 | 0 |
| enqueue | 36 | 36 | 24479968 | 33554432 | 0 | 255 | 0 | 0 | 0 |
| enqueue | 37 | 37 | 24484816 | 33554432 | 0 | 256 | 0 | 0 | 0 |
| enqueue | 38 | 38 | 24489712 | 33554432 | 0 | 257 | 0 | 0 | 0 |
| enqueue | 39 | 39 | 24494656 | 33554432 | 0 | 258 | 0 | 0 | 0 |
| enqueue | 40 | 40 | 24499648 | 33554432 | 0 | 259 | 0 | 0 | 0 |
| enqueue | 41 | 41 | 24504688 | 33554432 | 0 | 260 | 0 | 0 | 0 |
| enqueue | 42 | 42 | 24509776 | 33554432 | 0 | 261 | 0 | 0 | 0 |
| enqueue | 43 | 43 | 24514912 | 33554432 | 0 | 262 | 0 | 0 | 0 |
| enqueue | 44 | 44 | 24520096 | 33554432 | 0 | 263 | 0 | 0 | 0 |
| enqueue | 45 | 45 | 24525328 | 33554432 | 0 | 264 | 0 | 0 | 0 |
| enqueue | 46 | 46 | 24530608 | 33554432 | 0 | 265 | 0 | 0 | 0 |
| enqueue | 47 | 47 | 24535936 | 33554432 | 0 | 266 | 0 | 0 | 0 |
| enqueue | 48 | 48 | 24541312 | 33554432 | 0 | 267 | 0 | 0 | 0 |
| enqueue | 49 | 49 | 24546736 | 33554432 | 0 | 268 | 0 | 0 | 0 |
| enqueue | 50 | 50 | 24552208 | 33554432 | 0 | 269 | 0 | 0 | 0 |
| enqueue | 51 | 51 | 24557728 | 33554432 | 0 | 270 | 0 | 0 | 0 |
| enqueue | 52 | 52 | 24563296 | 33554432 | 0 | 271 | 0 | 0 | 0 |
| enqueue | 53 | 53 | 24568912 | 33554432 | 0 | 272 | 0 | 0 | 0 |
| enqueue | 54 | 54 | 24574576 | 33554432 | 0 | 273 | 0 | 0 | 0 |
| enqueue | 55 | 55 | 24580288 | 33554432 | 0 | 274 | 0 | 0 | 0 |
| enqueue | 56 | 56 | 24586048 | 33554432 | 0 | 275 | 0 | 0 | 0 |
| enqueue | 57 | 57 | 24591856 | 33554432 | 0 | 276 | 0 | 0 | 0 |
| enqueue | 58 | 58 | 24597712 | 33554432 | 0 | 277 | 0 | 0 | 0 |
| enqueue | 59 | 59 | 24603616 | 33554432 | 0 | 278 | 0 | 0 | 0 |
| enqueue | 60 | 60 | 24609568 | 33554432 | 0 | 279 | 0 | 0 | 0 |
| enqueue | 61 | 61 | 24615568 | 33554432 | 0 | 280 | 0 | 0 | 0 |
| enqueue | 62 | 62 | 24621616 | 33554432 | 0 | 281 | 0 | 0 | 0 |
| enqueue | 63 | 63 | 24627712 | 33554432 | 0 | 282 | 0 | 0 | 0 |
| enqueue | 64 | 64 | 24633856 | 33554432 | 0 | 283 | 0 | 0 | 0 |
| enqueue | 65 | 65 | 24640048 | 33554432 | 0 | 284 | 0 | 0 | 0 |
| enqueue | 66 | 66 | 24646288 | 33554432 | 0 | 285 | 0 | 0 | 0 |
| enqueue | 67 | 67 | 24652576 | 33554432 | 0 | 286 | 0 | 0 | 0 |
| enqueue | 68 | 68 | 24658912 | 33554432 | 0 | 287 | 0 | 0 | 0 |
| enqueue | 69 | 69 | 24665296 | 33554432 | 0 | 288 | 0 | 0 | 0 |
| enqueue | 70 | 70 | 24671728 | 33554432 | 0 | 289 | 0 | 0 | 0 |
| enqueue | 71 | 71 | 24678208 | 33554432 | 0 | 290 | 0 | 0 | 0 |
| enqueue | 72 | 0 | 24681280 | 33554432 | 0 | 291 | 0 | 0 | 0 |
| completion | — | — | 24681280 | 33554432 | 0 | 291 | 0 | 0 | 0 |
| drop | — | — | 24334336 | 33554432 | 0 | 291 | 0 | 0 | 0 |

### shape_stream_f64_n64_equal-shape-strides_counters (#1873 #1805, 1 threads)

Untimed public runtime observations; cumulative counters. Exact sequence and reset scope are recorded in the raw row.

| Phase | Index | Key | Live bytes | Pool reserved | Pending bytes | Allocations | Launches | Artifact entries | Module entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| before | — | — | 16914432 | 33554432 | 0 | 56 | 17 | 1 | 1 |
| enqueue | 0 | 0 | 16917504 | 33554432 | 0 | 57 | 17 | 1 | 1 |
| enqueue | 1 | 1 | 16920576 | 33554432 | 0 | 59 | 18 | 1 | 1 |
| enqueue | 2 | 0 | 16923648 | 33554432 | 0 | 60 | 18 | 1 | 1 |
| enqueue | 3 | 1 | 16926720 | 33554432 | 0 | 62 | 19 | 1 | 1 |
| enqueue | 4 | 0 | 16929792 | 33554432 | 0 | 63 | 19 | 1 | 1 |
| enqueue | 5 | 1 | 16932864 | 33554432 | 0 | 65 | 20 | 1 | 1 |
| enqueue | 6 | 0 | 16935936 | 33554432 | 0 | 66 | 20 | 1 | 1 |
| enqueue | 7 | 1 | 16939008 | 33554432 | 0 | 68 | 21 | 1 | 1 |
| enqueue | 8 | 0 | 16942080 | 33554432 | 0 | 69 | 21 | 1 | 1 |
| enqueue | 9 | 1 | 16945152 | 33554432 | 0 | 71 | 22 | 1 | 1 |
| enqueue | 10 | 0 | 16948224 | 33554432 | 0 | 72 | 22 | 1 | 1 |
| enqueue | 11 | 1 | 16951296 | 33554432 | 0 | 74 | 23 | 1 | 1 |
| enqueue | 12 | 0 | 16954368 | 33554432 | 0 | 75 | 23 | 1 | 1 |
| enqueue | 13 | 1 | 16957440 | 33554432 | 0 | 77 | 24 | 1 | 1 |
| enqueue | 14 | 0 | 16960512 | 33554432 | 0 | 78 | 24 | 1 | 1 |
| enqueue | 15 | 1 | 16963584 | 33554432 | 0 | 80 | 25 | 1 | 1 |
| enqueue | 16 | 0 | 16966656 | 33554432 | 0 | 81 | 25 | 1 | 1 |
| enqueue | 17 | 1 | 16969728 | 33554432 | 0 | 83 | 26 | 1 | 1 |
| enqueue | 18 | 0 | 16972800 | 33554432 | 0 | 84 | 26 | 1 | 1 |
| enqueue | 19 | 1 | 16975872 | 33554432 | 0 | 86 | 27 | 1 | 1 |
| enqueue | 20 | 0 | 16978944 | 33554432 | 0 | 87 | 27 | 1 | 1 |
| enqueue | 21 | 1 | 16982016 | 33554432 | 0 | 89 | 28 | 1 | 1 |
| enqueue | 22 | 0 | 16985088 | 33554432 | 0 | 90 | 28 | 1 | 1 |
| enqueue | 23 | 1 | 16988160 | 33554432 | 0 | 92 | 29 | 1 | 1 |
| enqueue | 24 | 0 | 16991232 | 33554432 | 0 | 93 | 29 | 1 | 1 |
| enqueue | 25 | 1 | 16994304 | 33554432 | 0 | 95 | 30 | 1 | 1 |
| enqueue | 26 | 0 | 16997376 | 33554432 | 0 | 96 | 30 | 1 | 1 |
| enqueue | 27 | 1 | 17000448 | 33554432 | 0 | 98 | 31 | 1 | 1 |
| enqueue | 28 | 0 | 17003520 | 33554432 | 0 | 99 | 31 | 1 | 1 |
| enqueue | 29 | 1 | 17006592 | 33554432 | 0 | 101 | 32 | 1 | 1 |
| enqueue | 30 | 0 | 17009664 | 33554432 | 0 | 102 | 32 | 1 | 1 |
| enqueue | 31 | 1 | 17012736 | 33554432 | 0 | 104 | 33 | 1 | 1 |
| completion | — | — | 17012736 | 33554432 | 0 | 104 | 33 | 1 | 1 |
| drop | — | — | 16914432 | 33554432 | 0 | 104 | 33 | 1 | 1 |


## State diagnostics (no timing)

Allocation and lifetime state, read before and after each arm: driver free device bytes for the pool, and process RSS where the issue's observable is host memory (#1924). Bytes, not nanoseconds; a case that keeps memory it no longer needs shows up as a non-zero retained or growth value, and a probe that the driver refuses shows up in `probe_status`.

### vendor_churn_f64_cycles8 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 8 | — |
| free_bytes_after | 83901153280 | 80014.4 |
| free_bytes_before | 83901153280 | 80014.4 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 781529088 | 745.3 |
| rss_bytes_before | 781082624 | 744.9 |
| rss_growth_bytes | 446464 | 0.4 |
| case_process_and_validation_wall_ns | 5278244266 | — |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 83901153280 | 781529088 |
| 1 | 0 | 83901153280 | 781529088 |
| 2 | 0 | 83901153280 | 781529088 |
| 3 | 0 | 83901153280 | 781529088 |
| 4 | 0 | 83901153280 | 781529088 |
| 5 | 0 | 83901153280 | 781529088 |
| 6 | 0 | 83901153280 | 781529088 |
| 7 | 0 | 83901153280 | 781529088 |

### vendor_churn_f64_cycles64 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 64 | — |
| free_bytes_after | 83901153280 | 80014.4 |
| free_bytes_before | 83901153280 | 80014.4 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 781598720 | 745.4 |
| rss_bytes_before | 781156352 | 745.0 |
| rss_growth_bytes | 442368 | 0.4 |
| case_process_and_validation_wall_ns | 6439347050 | — |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 83901153280 | 781594624 |
| 1 | 0 | 83901153280 | 781594624 |
| 2 | 0 | 83901153280 | 781594624 |
| 3 | 0 | 83901153280 | 781594624 |
| 4 | 0 | 83901153280 | 781594624 |
| 5 | 0 | 83901153280 | 781594624 |
| 6 | 0 | 83901153280 | 781594624 |
| 7 | 0 | 83901153280 | 781594624 |
| 8 | 0 | 83901153280 | 781594624 |
| 9 | 0 | 83901153280 | 781594624 |
| 10 | 0 | 83901153280 | 781594624 |
| 11 | 0 | 83901153280 | 781598720 |
| 12 | 0 | 83901153280 | 781598720 |
| 13 | 0 | 83901153280 | 781598720 |
| 14 | 0 | 83901153280 | 781598720 |
| 15 | 0 | 83901153280 | 781598720 |
| 16 | 0 | 83901153280 | 781598720 |
| 17 | 0 | 83901153280 | 781598720 |
| 18 | 0 | 83901153280 | 781598720 |
| 19 | 0 | 83901153280 | 781598720 |
| 20 | 0 | 83901153280 | 781598720 |
| 21 | 0 | 83901153280 | 781598720 |
| 22 | 0 | 83901153280 | 781598720 |
| 23 | 0 | 83901153280 | 781598720 |
| 24 | 0 | 83901153280 | 781598720 |
| 25 | 0 | 83901153280 | 781598720 |
| 26 | 0 | 83901153280 | 781598720 |
| 27 | 0 | 83901153280 | 781598720 |
| 28 | 0 | 83901153280 | 781598720 |
| 29 | 0 | 83901153280 | 781598720 |
| 30 | 0 | 83901153280 | 781598720 |
| 31 | 0 | 83901153280 | 781598720 |
| 32 | 0 | 83901153280 | 781598720 |
| 33 | 0 | 83901153280 | 781598720 |
| 34 | 0 | 83901153280 | 781598720 |
| 35 | 0 | 83901153280 | 781598720 |
| 36 | 0 | 83901153280 | 781598720 |
| 37 | 0 | 83901153280 | 781598720 |
| 38 | 0 | 83901153280 | 781598720 |
| 39 | 0 | 83901153280 | 781598720 |
| 40 | 0 | 83901153280 | 781598720 |
| 41 | 0 | 83901153280 | 781598720 |
| 42 | 0 | 83901153280 | 781598720 |
| 43 | 0 | 83901153280 | 781598720 |
| 44 | 0 | 83901153280 | 781598720 |
| 45 | 0 | 83901153280 | 781598720 |
| 46 | 0 | 83901153280 | 781598720 |
| 47 | 0 | 83901153280 | 781598720 |
| 48 | 0 | 83901153280 | 781598720 |
| 49 | 0 | 83901153280 | 781598720 |
| 50 | 0 | 83901153280 | 781598720 |
| 51 | 0 | 83901153280 | 781598720 |
| 52 | 0 | 83901153280 | 781598720 |
| 53 | 0 | 83901153280 | 781598720 |
| 54 | 0 | 83901153280 | 781598720 |
| 55 | 0 | 83901153280 | 781598720 |
| 56 | 0 | 83901153280 | 781598720 |
| 57 | 0 | 83901153280 | 781598720 |
| 58 | 0 | 83901153280 | 781598720 |
| 59 | 0 | 83901153280 | 781598720 |
| 60 | 0 | 83901153280 | 781598720 |
| 61 | 0 | 83901153280 | 781598720 |
| 62 | 0 | 83901153280 | 781598720 |
| 63 | 0 | 83901153280 | 781598720 |

### pool_retention_f64_alloc-free (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 83999719424 | 80108.4 |
| free_bytes_after_recreate | 83999719424 | 80108.4 |
| free_bytes_start | 83999719424 | 80108.4 |
| free_bytes_while_alive | 75409784832 | 71916.4 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (alloc-free arm)` | — |
| retained_bytes_after_drop | 0 | 0.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |
| case_process_and_validation_wall_ns | 4207723857 | — |

### pool_retention_f64_runtime-recreate (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 83999719424 | 80108.4 |
| free_bytes_after_recreate | 83999719424 | 80108.4 |
| free_bytes_start | 83999719424 | 80108.4 |
| free_bytes_while_alive | 75409784832 | 71916.4 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (runtime-recreate arm)` | — |
| retained_bytes_after_drop | 0 | 0.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |
| case_process_and_validation_wall_ns | 4017827289 | — |

### pool_retention_f64_driver-alloc-after-free (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 83999719424 | 80108.4 |
| free_bytes_after_recreate | 83999719424 | 80108.4 |
| free_bytes_start | 83999719424 | 80108.4 |
| free_bytes_while_alive | 75409784832 | 71916.4 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `ok (6442450944 bytes)` | — |
| retained_bytes_after_drop | 0 | 0.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |
| case_process_and_validation_wall_ns | 4085749885 | — |


## First-call diagnostics (not steady-state rows)

One cold process; each distinct layout is uploaded outside timing, then its first and second call are timed separately (each includes a device synchronize).

### first_call_transpose_to-contiguous_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.022 ms, second call 0.022 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 51091.259 | 0.032 |
| [2, 5] -> transpose |  | 0.024 | 0.021 |
| [2, 7] -> transpose |  | 0.025 | 0.021 |
| [3, 5] -> transpose |  | 0.022 | 0.022 |
| [3, 7] -> transpose |  | 0.022 | 0.021 |
| [4, 5] -> transpose |  | 0.022 | 0.030 |
| [4, 7] -> transpose |  | 0.022 | 0.022 |
| [5, 5] -> transpose |  | 0.022 | 0.021 |
| [5, 7] -> transpose |  | 0.022 | 0.021 |
| [6, 5] -> transpose |  | 0.022 | 0.024 |
| [6, 7] -> transpose |  | 0.023 | 0.022 |
| [7, 5] -> transpose |  | 0.021 | 0.021 |
| [7, 7] -> transpose |  | 0.022 | 0.021 |
| [8, 5] -> transpose |  | 0.021 | 0.022 |
| [8, 7] -> transpose |  | 0.022 | 0.030 |
| [9, 5] -> transpose |  | 0.022 | 0.022 |
| [9, 7] -> transpose |  | 0.021 | 0.021 |
| [10, 5] -> transpose |  | 0.022 | 0.021 |
| [10, 7] -> transpose |  | 0.022 | 0.022 |
| [11, 5] -> transpose |  | 0.022 | 0.021 |
| [11, 7] -> transpose |  | 0.023 | 0.021 |
| [12, 5] -> transpose |  | 0.022 | 0.022 |
| [12, 7] -> transpose |  | 0.022 | 0.022 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.020 ms, second call 0.018 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 14.529 | 0.026 |
| [2, 5] -> transpose |  | 0.021 | 0.018 |
| [2, 7] -> transpose |  | 0.022 | 0.018 |
| [3, 5] -> transpose |  | 0.020 | 0.019 |
| [3, 7] -> transpose |  | 0.020 | 0.018 |
| [4, 5] -> transpose |  | 0.020 | 0.018 |
| [4, 7] -> transpose |  | 0.020 | 0.018 |
| [5, 5] -> transpose |  | 0.020 | 0.018 |
| [5, 7] -> transpose |  | 0.021 | 0.018 |
| [6, 5] -> transpose |  | 0.020 | 0.018 |
| [6, 7] -> transpose |  | 0.020 | 0.019 |
| [7, 5] -> transpose |  | 0.020 | 0.019 |
| [7, 7] -> transpose |  | 0.021 | 0.022 |
| [8, 5] -> transpose |  | 0.020 | 0.018 |
| [8, 7] -> transpose |  | 0.021 | 0.019 |
| [9, 5] -> transpose |  | 0.022 | 0.018 |
| [9, 7] -> transpose |  | 0.020 | 0.018 |
| [10, 5] -> transpose |  | 0.020 | 0.028 |
| [10, 7] -> transpose |  | 0.020 | 0.018 |
| [11, 5] -> transpose |  | 0.020 | 0.018 |
| [11, 7] -> transpose |  | 0.020 | 0.018 |
| [12, 5] -> transpose |  | 0.020 | 0.018 |
| [12, 7] -> transpose |  | 0.020 | 0.019 |


## Timing boundaries

- **alloc_zero / alloc-zero-per-callback** — inside: with_raw entry/exit per allocation, alloc_output (zero-filled), device synchronize at batch end; outside: backend construction, backend session entry/exit, initialization allocation and its raw-scope entry/exit (within-callback arm), shared raw execution scope entry/exit for the measured batches (within-callback arm), output-retention container allocation, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: alloc_output (zero-filled) x iterations, stream synchronize at batch end; outside: backend construction, backend session entry/exit, initialization allocation and its raw-scope entry/exit (within-callback arm), shared raw execution scope entry/exit for the measured batches (within-callback arm), output-retention container allocation, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, initialization allocation and its raw-scope entry/exit (within-callback arm), shared raw execution scope entry/exit for the measured batches (within-callback arm), output-retention container allocation, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **borrowed_input_gemm / owned-input** — inside: one small dot_general_read with owned inputs, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **borrowed_input_gemm / view-input** — inside: one small dot_general_read with a borrowed transposed input view, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **cholesky_staging / direct** — inside: one public Cholesky call including intrinsic solver-info readback, output allocation, batch-edge device synchronization; outside: HPD fixture and upload, engine, eager runtime and session setup, EagerTensor wrapping, explicit first call and full reconstruction, output download and destruction.
- **cholesky_staging / eager** — inside: one public Cholesky call including intrinsic solver-info readback, output allocation, batch-edge device synchronization; outside: HPD fixture and upload, engine, eager runtime and session setup, EagerTensor wrapping, explicit first call and full reconstruction, output download and destruction.
- **cholesky_staging / pytorch** — inside: one public PyTorch CUDA operation, batch-edge device synchronization; outside: fixture, upload and layout conversion, lazy initialization, validation and download, retention allocation and output destruction.
- **contraction_plan_churn / below-capacity-varying-workspace** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / distinct-plans** — inside: one re-run of the first contraction after the plan cache was filled by distinct shapes, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / repeated-plan** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **eager_op / eager-per-call** — inside: one eager op with a with_eager_session entry/exit per operation, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session** — inside: one eager op in a reused with_eager_session, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session-streaming** — inside: one eager op in a reused with_eager_session, result dropped per op, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / launch-only-floor** — inside: one add_read_into on a reused output (no per-op allocation), one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / divide-and-conquer** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / divide-and-conquer** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / jacobi** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / jacobi** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **first_call_layouts / copy-into** — inside: to_contiguous_view (or copy_read_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **first_call_layouts / to-contiguous** — inside: to_contiguous_view (or copy_read_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **gather_table / cudarc-row-copies** — inside: 8 device-to-device row copies (cudarc) into a freshly allocated destination, stream.synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **gather_table / eager-gather** — inside: one eager gather of 8 indices from a [vocab, hidden] f32 device table, one device synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **hadamard_dot_general / dot-general** — inside: one dot_general with every axis as a batch axis (pure Hadamard), device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **hadamard_dot_general / multiply** — inside: one elementwise multiply, device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **linalg_readback_latency / direct** — inside: one CUDA solve or QR of an n x n f64 device matrix, dispatch and batch-edge device sync, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, residual or reconstruction check (download).
- **linalg_readback_latency / pytorch** — inside: one public PyTorch CUDA operation, batch-edge device synchronization; outside: fixture, upload and layout conversion, lazy initialization, validation and download, retention allocation and output destruction.
- **owned_destination / borrowed-dst** — inside: dot_general_read_into a reused borrowed-view destination, one isolated call per sample; device synchronize after the call; outside: backend construction, operand and destination upload, the destination handle built per call, correctness check (download).
- **owned_destination / owned-dst** — inside: dot_general_read_into a reused owned destination, one isolated call per sample; device synchronize after the call; outside: backend construction, operand and destination upload, the destination handle built per call, correctness check (download).
- **pool_retention / alloc-free** — inside: —; outside: everything (a state diagnostic, no timing).
- **pool_retention / driver-alloc-after-free** — inside: —; outside: everything (a state diagnostic, no timing).
- **pool_retention / runtime-recreate** — inside: —; outside: everything (a state diagnostic, no timing).
- **prepared_decode / eager-session** — inside: one prepared Runtime::run or eager dot_general, batch-edge device synchronization; outside: fixture and upload, engine and eager session creation, trace compile and prepare, validation and output inspection, output destruction.
- **prepared_decode / prepared** — inside: one prepared Runtime::run or eager dot_general, batch-edge device synchronization; outside: fixture and upload, engine and eager session creation, trace compile and prepare, validation and output inspection, output destruction.
- **prepared_decode / pytorch** — inside: one public PyTorch CUDA operation, batch-edge device synchronization; outside: fixture, upload and layout conversion, lazy initialization, validation and download, retention allocation and output destruction.
- **raw_scope_entry / empty-behind-queued-gemm** — inside: the second empty with_raw scope after queued device work; outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **raw_scope_entry / empty-idle** — inside: with_cuda_engine + empty with_raw closure on an idle device; outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **raw_scope_entry / sync-behind-queued-gemm** — inside: explicit device synchronize behind queued device work (the reference barrier); outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **reshape_materialize / constant** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
- **reshape_materialize / op-output** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
- **shape_stream / capacity-thrash** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / equal-shape-strides** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / fixed-warm** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / fresh** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / mixed-warm** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **small_blocks / batched** — inside: one batched dot_general_read ([n, n, count]), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **small_blocks / loop** — inside: count x dot_general_read (one submission per block), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **standard_batched_matmul / batch-outer** — inside: one allocating batched dot_general_read, batch-edge device synchronization; outside: input generation and layout conversion, engine creation and upload, full scalar oracle and download, output destruction.
- **standard_batched_matmul / batch-trailing** — inside: one allocating batched dot_general_read, batch-edge device synchronization; outside: input generation and layout conversion, engine creation and upload, full scalar oracle and download, output destruction.
- **standard_batched_matmul / pytorch** — inside: one public PyTorch CUDA operation, batch-edge device synchronization; outside: fixture, upload and layout conversion, lazy initialization, validation and download, retention allocation and output destruction.
- **svd_driver / auto** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / jacobi** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / polar** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / qr-based** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **transfer / cudarc-pageable** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / cudarc-pageable-fresh** — inside: cudarc memcpy_dtoh into a freshly allocated Vec per call, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / cudarc-pinned** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: download_tensor (synchronizes itself), host output allocation; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: upload_tensor, runtime.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro-into** — inside: download_pending into one caller-owned pinned host buffer reused across calls, the pending download's wait synchronizes itself; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **vendor_churn / backend-cycle** — inside: —; outside: everything (a state diagnostic, no timing).
- **vendor_handle / create-destroy** — inside: cublasCreate + cublasSetStream + dgemm + cublasDestroy inside one with_raw scope per operation, host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.
- **vendor_handle / reused** — inside: cublasSetStream + dgemm inside one with_raw scope per operation (handle reused), host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.

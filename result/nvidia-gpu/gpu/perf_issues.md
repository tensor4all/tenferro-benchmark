# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261017_110340`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

## run_t1

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261017_110340/run_t1.yaml`

```yaml
timestamp: '2026-10-17T11:03:42Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/extern/tenferro-rs
  commit: 88daa885a5bb09e07f3031814bbb1a21f7a74e62
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/extern/tenferro-rs
blas:
  implementation: none
environment:
  hostname: 60145713b7ef
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: 5190758f49f1d17be2c6dbcd8ccb0cfe69b9be1c
    OPENBLAS_ROOT: /opt/openblas
    CUDA_HOME: /usr/local/cuda
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    USE_CUDA: '1'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
    CUDA_VISIBLE_DEVICES: GPU-530977e1-4968-9283-4129-9fbec3e66542
    GPU_BENCH_PHYSICAL_DEVICE: '0'
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 6, coverage `quick`, effort `standard`
- Expected 139, selected 139, executed 138, unsupported 0, failed 1, missing 0, noisy 5.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Failed: `eager_op_f32_dot_general_decode_launch-only-floor@t1`
- Noisy: `transfer_up_f64_tenferro_8B@t1`, `transfer_up_f64_tenferro_4096B@t1`, `alloc_zero_f64_host-zero-upload_16el@t1`, `alloc_zero_f64_host-zero-upload_512el@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`

Numerical checks: **138/139 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 35780.8 | 7511.9 | 15.3% | NOISY |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6557.8 | 60.4 | 1.1% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9834.7 | 63.0 | 1.3% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 39827.9 | 7314.1 | 13.2% | NOISY |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7204.3 | 44.3 | 0.9% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 60200.3 | 196.8 | 0.3% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 32 | 96360.0 | 4458.6 | 4.0% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13107.3 | 71.3 | 0.4% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12657.3 | 32.4 | 0.5% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 380703.6 | 2940.1 | 1.4% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 90460.1 | 378.6 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50498.7 | 124.8 | 0.3% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1230133.0 | 53502.2 | 4.1% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 756564.8 | 1156.1 | 0.1% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636782.5 | 198.8 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 22648069.0 | 102743.5 | 0.4% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 14004049.0 | 295091.5 | 1.9% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10018303.0 | 270.0 | 0.0% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 15708.5 | 105.6 | 0.5% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 7790.6 | 31.3 | 0.6% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9198.6 | 95.4 | 0.7% | ok |
| transfer_down_f64_tenferro-into_8B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 22064.0 | 64.5 | 0.4% | ok |
| transfer_down_f64_cudarc-pageable-fresh_8B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7342.5 | 14.6 | 0.4% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 22746.0 | 1351.3 | 3.9% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8328.3 | 117.9 | 1.5% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9374.5 | 84.4 | 1.2% | ok |
| transfer_down_f64_tenferro-into_4096B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 21938.9 | 214.2 | 2.1% | ok |
| transfer_down_f64_cudarc-pageable-fresh_4096B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8324.0 | 381.9 | 2.7% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 65118.3 | 254.5 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 13071.1 | 489.3 | 2.4% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11676.7 | 68.4 | 0.6% | ok |
| transfer_down_f64_tenferro-into_65536B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 128 | 28372.5 | 103.3 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable-fresh_65536B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=65536, direction=down | 1 | 64 | 45926.6 | 5578.6 | 6.2% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 657605.8 | 56502.2 | 5.9% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 94525.2 | 390.8 | 0.4% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49721.3 | 73.8 | 0.1% | ok |
| transfer_down_f64_tenferro-into_1048576B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 32 | 114371.3 | 257.0 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable-fresh_1048576B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=1048576, direction=down | 1 | 8 | 651595.5 | 1098.1 | 0.2% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 9617376.0 | 183387.2 | 2.1% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 854983.5 | 3246.5 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 646727.8 | 385.0 | 0.3% | ok |
| transfer_down_f64_tenferro-into_16777216B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 1055521.5 | 6185.5 | 0.5% | ok |
| transfer_down_f64_cudarc-pageable-fresh_16777216B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=16777216, direction=down | 1 | 1 | 2092861.0 | 9350.0 | 0.4% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 143944133.0 | 3700340.5 | 3.0% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 27708792.0 | 1980252.0 | 5.3% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10197227.0 | 9275.5 | 1.2% | ok |
| transfer_down_f64_tenferro-into_268435456B | #2009 | tenferro-into | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 16053636.0 | 628900.0 | 3.1% | ok |
| transfer_down_f64_cudarc-pageable-fresh_268435456B | #2009 | cudarc-pageable-fresh | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 145787209.0 | 6862692.0 | 3.1% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 3000.3 | 17.0 | 0.6% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 512 | 4087.4 | 57.1 | 1.1% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 256 | 10599.1 | 219.0 | 1.4% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 256 | 28795.9 | 14353.2 | 33.2% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 256 | 11758.9 | 219.1 | 1.4% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 256 | 27270.8 | 13844.0 | 33.2% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 13644.8 | 864.5 | 5.3% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 8 | 172255.4 | 1168.2 | 12.3% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 16395.6 | 298.1 | 1.6% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1187261.5 | 34923.8 | 4.0% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 5069617.0 | 200482.5 | 5.3% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 22877.0 | 382.9 | 1.2% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 4870339.0 | 224140.0 | 8.5% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 24546.1 | 332.7 | 1.0% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 4676002.0 | 307661.5 | 3.9% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 64 | 25453.5 | 294.0 | 1.0% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 5760613.0 | 20627.0 | 1.7% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 32 | 25617.2 | 237.5 | 0.6% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5329197.0 | 22611.0 | 5.0% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2387703.0 | 7790.0 | 0.4% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 91873497.0 | 149615.5 | 0.3% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35145209.0 | 34631.0 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 16836492.0 | 71992.5 | 0.7% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14306214.0 | 76408.0 | 0.6% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 285038467.0 | 6034436.5 | 1.3% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 225899632.0 | 251419.5 | 0.2% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 332249.1 | 1431.3 | 0.5% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 345640.9 | 11408.5 | 2.4% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 334180.4 | 6301.5 | 1.8% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 9916.2 | 150.6 | 1.7% | ok |
| eager_op_f32_add_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 9399.1 | 85.5 | 0.7% | ok |
| eager_op_f32_add_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 6464.7 | 32.1 | 2.6% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 29262.8 | 663.9 | 2.0% | ok |
| eager_op_f32_dot_general_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 29577.9 | 1161.4 | 4.0% | ok |
| eager_op_f32_dot_general_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | — | — | — | — | FAILED |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 256 | 13882.8 | 692.4 | 4.5% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 28900.5 | 104.8 | 1.5% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 256 | 13781.6 | 893.6 | 4.5% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 29071.0 | 73.9 | 0.5% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 13588.3 | 1402.8 | 5.7% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 29611.9 | 787.3 | 1.9% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 332490.2 | 778.8 | 0.2% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 333307.8 | 1708.8 | 0.3% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4685620.0 | 33010.5 | 0.9% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4648098.0 | 6730.0 | 0.1% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6295520.0 | 11605.5 | 0.1% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3504004.0 | 5706.0 | 0.2% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41025569.0 | 16041.0 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41081771.0 | 34071.5 | 0.1% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32292415.0 | 12521.0 | 0.1% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9149050.0 | 13741.0 | 0.1% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 387827386.0 | 95380.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 387416933.0 | 50383.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 258787258.0 | 155963.5 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 61664307.0 | 123750.0 | 0.1% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 309161.8 | 7445.9 | 1.4% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 284764.6 | 808.8 | 0.9% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 199183.2 | 3209.5 | 1.0% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 435572.9 | 3320.1 | 0.6% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 438266.6 | 8069.1 | 1.3% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 392452.5 | 11779.2 | 2.2% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 2 | 1957889.5 | 8400.8 | 0.8% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 2 | 1953239.0 | 4127.2 | 0.2% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19759212.0 | 22996.0 | 0.3% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 314867.0 | 2616.2 | 3.3% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 312138.1 | 936.9 | 0.3% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 277740.6 | 923.2 | 0.2% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 469307.9 | 7545.3 | 1.3% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 469151.5 | 1265.0 | 1.0% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 652837.2 | 7456.8 | 0.7% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3197181.0 | 6530.0 | 0.2% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3178380.0 | 4670.0 | 1.3% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49259692.0 | 83463.0 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 32 | 17336.0 | 769.5 | 4.3% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 10769.0 | 147.1 | 1.5% | ok |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 256 | 11669.3 | 177.4 | 1.0% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 256 | 13315.5 | 339.4 | 2.3% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 18119.5 | 238.4 | 1.1% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 20615.6 | 306.1 | 1.1% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 18932.5 | 87.5 | 5.0% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 18534.3 | 80.1 | 1.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 20429.3 | 822.0 | 4.5% | ok |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 21817.5 | 1279.7 | 3.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 23357.2 | 330.8 | 1.0% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 23182.2 | 413.9 | 1.4% | ok |

## Whole-sequence CUDA shape-stream latency

One sample is the entire recorded sequence, including required planning and one final native synchronization. Cache reset and explicitly declared key warmup are outside timing. Median/IQR describe sequence latency, not single-operation throughput. Enqueue durations and the exact key/shape/stride sequence are retained in raw rows. Counter observations run separately. NOISY is descriptive; these screens establish no baseline/candidate performance claim.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Sequences/sample | Median ns/sequence | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| shape_stream_f64_n16_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 714487.0 | 12976.0 | 2.0% | ok |
| shape_stream_f64_n16_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 719797.0 | 10500.0 | 1.1% | ok |
| shape_stream_f64_n16_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 316882.0 | 12455.0 | 2.9% | ok |
| shape_stream_f64_n64_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 762279.0 | 13106.0 | 0.9% | ok |
| shape_stream_f64_n64_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 743889.0 | 7765.0 | 0.9% | ok |
| shape_stream_f64_n64_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 380234.0 | 5494.5 | 1.7% | ok |

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


## Timing boundaries

- **alloc_zero / alloc-fill-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_output + fill_zero_write (device fill), device synchronize at batch end; outside: backend construction, backend session entry/exit, shared raw execution scope entry/exit for within-callback arm, output-retention container allocation, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_zero_output, device synchronize at batch end; outside: backend construction, backend session entry/exit, shared raw execution scope entry/exit for within-callback arm, output-retention container allocation, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: alloc_zero_output x iterations, device synchronize at batch end; outside: backend construction, backend session entry/exit, shared raw execution scope entry/exit for within-callback arm, output-retention container allocation, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, shared raw execution scope entry/exit for within-callback arm, output-retention container allocation, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **eager_op / eager-session** — inside: one eager op in a reused with_eager_session, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session-streaming** — inside: one eager op in a reused with_eager_session, result dropped per op, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / launch-only-floor** — inside: one add_read_into on a reused output (no per-op allocation), one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevd** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevd** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevj** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevj** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **gather_table / cudarc-row-copies** — inside: 8 device-to-device row copies (cudarc) into a freshly allocated destination, stream.synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **gather_table / eager-gather** — inside: one eager gather of 8 indices from a [vocab, hidden] f32 device table, one device synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **hadamard_dot_general / dot-general** — inside: one dot_general with every axis as a batch axis (pure Hadamard), device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **hadamard_dot_general / multiply** — inside: one elementwise multiply, device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **reshape_materialize / constant** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
- **reshape_materialize / op-output** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
- **shape_stream / fixed-warm** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / fresh** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **shape_stream / mixed-warm** — inside: entire declared shape sequence, including required planning, then native synchronization; outside: backend/session/library initialization, input generation/upload and read-handle preparation, controlled CUDA extension-cache reset and declared warm-key precondition, full-output download/verification and output destruction.
- **small_blocks / batched** — inside: one batched dot_general_read ([n, n, count]), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **small_blocks / loop** — inside: count x dot_general_read (one submission per block), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **svd_driver / auto** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / gesvd** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / gesvdj** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **svd_driver / xgesvdp** — inside: one thin SVD of an n x n c64 device matrix with an explicit cuSOLVER driver, device synchronize at the batch edge; outside: backend construction, host matrix construction and upload, reconstruction check (download).
- **transfer / cudarc-pageable** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / cudarc-pageable-fresh** — inside: cudarc memcpy_dtoh into a freshly allocated Vec per call, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / cudarc-pinned** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: download_tensor (synchronizes itself), host output allocation; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: upload_tensor, runtime.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro-into** — inside: download_tensor_into into one caller-owned Vec reused across calls, the copy synchronizes itself; outside: backend/context construction, host and device buffer allocation, round-trip check.

### FAILED: eager_op_f32_dot_general_decode_launch-only-floor (1 threads)
```
Error: "unknown #2079 floor op dot_general"
```

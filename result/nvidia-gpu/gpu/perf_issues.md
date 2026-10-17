# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261017_002510`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261017_002510/run_t1.yaml`

```yaml
timestamp: '2026-10-17T00:25:12Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/extern/tenferro-rs
  commit: 2754c8c85dcc35c1b63cb7bbdd94ddfd7ea5746e
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/extern/tenferro-rs
blas:
  implementation: none
environment:
  hostname: 4ba652984ab5
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: f94171e1f76a246bdaee4f2e008ffc74149a6d5f
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

- Manifest: `gpu/perf_issues` version 5, coverage `quick`, effort `standard`
- Expected 127, selected 127, executed 126, unsupported 0, failed 1, missing 0, noisy 4.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Failed: `eager_op_f32_dot_general_decode_launch-only-floor@t1`
- Noisy: `transfer_up_f64_tenferro_8B@t1`, `alloc_zero_f64_host-zero-upload_16el@t1`, `alloc_zero_f64_host-zero-upload_512el@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`

Numerical checks: **126/127 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 33972.2 | 6882.7 | 14.7% | NOISY |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6362.2 | 60.6 | 1.1% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 10341.9 | 131.9 | 1.2% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 64 | 34912.5 | 3495.3 | 9.2% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7232.5 | 280.0 | 2.4% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 61266.9 | 70.1 | 0.1% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 32 | 96829.2 | 2939.3 | 3.5% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13812.6 | 130.1 | 0.6% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12643.9 | 363.2 | 2.6% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 351380.4 | 9001.6 | 2.8% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 89355.5 | 14029.1 | 8.0% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50072.5 | 218.4 | 0.3% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1182793.0 | 27838.2 | 2.6% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 747877.5 | 5210.2 | 0.7% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636455.8 | 348.6 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 21804081.0 | 209823.0 | 0.6% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 19125972.0 | 270379.0 | 4.7% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10016017.0 | 554.0 | 0.0% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 256 | 15346.6 | 416.9 | 2.9% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7479.7 | 15.7 | 0.3% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 8648.0 | 124.5 | 1.4% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 23385.4 | 360.5 | 2.8% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8035.1 | 44.4 | 0.4% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9271.0 | 335.7 | 2.6% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 60706.3 | 348.6 | 2.7% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 13696.6 | 50.7 | 0.7% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11318.3 | 145.1 | 1.1% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 585972.8 | 2595.8 | 2.4% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 94416.3 | 1493.5 | 1.4% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49409.5 | 107.0 | 0.3% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 9152491.0 | 437343.8 | 3.0% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 1111490.5 | 2415.0 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 645628.8 | 163.8 | 0.2% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 142633473.0 | 4225615.0 | 1.9% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 14158790.0 | 11080.0 | 0.1% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10185984.0 | 8050.0 | 0.8% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 2803.5 | 61.6 | 4.0% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 512 | 4336.1 | 58.1 | 1.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 256 | 10785.2 | 160.4 | 2.2% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 256 | 27274.4 | 14173.9 | 35.1% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 256 | 11277.6 | 193.4 | 1.8% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 256 | 23872.6 | 11913.9 | 32.7% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 11719.8 | 204.5 | 5.0% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 8 | 157737.1 | 683.1 | 10.3% | NOISY |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 15869.7 | 560.8 | 2.7% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1159212.5 | 2242.5 | 0.2% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 4438423.0 | 12775.0 | 0.2% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 21914.3 | 497.2 | 1.4% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 4658121.0 | 291306.0 | 5.8% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 21160.9 | 215.4 | 0.6% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 4781926.0 | 270460.0 | 6.4% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 64 | 24211.2 | 1021.1 | 3.9% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 5406478.0 | 344793.0 | 6.5% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 64 | 23262.0 | 1067.1 | 2.7% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5243432.0 | 281915.5 | 3.8% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2400329.0 | 7135.0 | 0.2% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 93779840.0 | 705961.5 | 0.5% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35499852.0 | 5985.5 | 0.0% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 16883399.0 | 43996.5 | 0.4% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14174270.0 | 44351.5 | 0.2% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 284026088.0 | 2863230.0 | 0.6% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 225841844.0 | 581711.5 | 0.3% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 332541.0 | 3098.9 | 1.0% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 338204.9 | 1378.9 | 0.4% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 337468.6 | 1625.0 | 0.7% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 9875.0 | 111.6 | 0.7% | ok |
| eager_op_f32_add_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 10453.6 | 392.9 | 3.1% | ok |
| eager_op_f32_add_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 6751.4 | 173.8 | 2.4% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 29145.0 | 421.1 | 1.4% | ok |
| eager_op_f32_dot_general_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 31680.5 | 1928.0 | 5.4% | ok |
| eager_op_f32_dot_general_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | — | — | — | — | FAILED |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 13800.7 | 1169.1 | 5.4% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 25515.6 | 171.7 | 0.4% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 256 | 14000.2 | 307.9 | 2.0% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 29304.5 | 243.1 | 0.6% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 256 | 12739.4 | 94.3 | 0.6% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 25524.3 | 242.8 | 0.9% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 332647.2 | 1271.4 | 0.3% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 332292.2 | 622.5 | 0.2% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4655811.0 | 9250.5 | 0.3% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4691292.0 | 25946.0 | 0.5% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6418265.0 | 13870.0 | 0.5% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3523389.0 | 12965.5 | 0.2% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41070716.0 | 16481.0 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41118248.0 | 27811.5 | 0.1% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 31828487.0 | 57797.0 | 0.2% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9150026.0 | 18805.0 | 0.1% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 387993415.0 | 186682.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 387514437.0 | 143950.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 259437941.0 | 140385.0 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 61509595.0 | 123019.5 | 0.1% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 293085.8 | 7630.4 | 2.9% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 316740.4 | 1788.1 | 0.4% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 210820.8 | 1401.3 | 1.1% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 420481.6 | 12730.5 | 1.8% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 431013.2 | 8346.6 | 1.1% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 382265.2 | 1622.6 | 0.3% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 2 | 1950436.5 | 6592.8 | 0.3% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 2 | 1947071.5 | 5857.8 | 0.7% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19659801.0 | 15800.5 | 0.3% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 315822.8 | 1432.6 | 0.3% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 310980.1 | 1853.2 | 0.4% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 278914.0 | 3496.4 | 1.1% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 478876.2 | 5953.3 | 0.9% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 468005.9 | 2844.4 | 0.7% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 620187.8 | 1805.0 | 0.8% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3203258.0 | 3885.0 | 0.1% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3173987.0 | 6545.5 | 0.2% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49166183.0 | 42516.5 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 10798.2 | 204.9 | 5.5% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 10298.0 | 392.7 | 8.2% | ok |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 256 | 11912.7 | 157.2 | 2.3% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 256 | 11117.8 | 238.0 | 1.8% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 18515.4 | 564.2 | 3.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 18945.0 | 254.8 | 0.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 20522.5 | 607.4 | 2.8% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 20063.8 | 526.2 | 3.0% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 19383.5 | 183.1 | 3.5% | ok |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 19574.7 | 923.9 | 4.4% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 23733.0 | 616.0 | 1.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 21003.3 | 193.4 | 7.3% | ok |

## Whole-sequence CUDA shape-stream latency

One sample is the entire recorded sequence, including required planning and one final native synchronization. Cache reset and explicitly declared key warmup are outside timing. Median/IQR describe sequence latency, not single-operation throughput. Enqueue durations and the exact key/shape/stride sequence are retained in raw rows. Counter observations run separately. NOISY is descriptive; these screens establish no baseline/candidate performance claim.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Sequences/sample | Median ns/sequence | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| shape_stream_f64_n16_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 692125.0 | 11255.5 | 1.5% | ok |
| shape_stream_f64_n16_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 578812.0 | 10510.5 | 1.2% | ok |
| shape_stream_f64_n16_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=16, instrument=False | 1 | 1 | 319802.0 | 2690.5 | 1.8% | ok |
| shape_stream_f64_n64_fixed-warm_latency | #1873 #1805 | fixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 675774.0 | 9380.0 | 1.1% | ok |
| shape_stream_f64_n64_mixed-warm_latency | #1873 #1805 | mixed-warm | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 689145.0 | 7615.0 | 1.1% | ok |
| shape_stream_f64_n64_fresh_latency | #1873 #1805 | fresh | tenferro-cuda | f64 | n=64, instrument=False | 1 | 1 | 347643.0 | 7766.0 | 1.4% | ok |

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
- **transfer / cudarc-pinned** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: download_tensor (synchronizes itself), host output allocation; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: upload_tensor, runtime.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.

### FAILED: eager_op_f32_dot_general_decode_launch-only-floor (1 threads)
```
Error: "unknown #2079 floor op dot_general"
```

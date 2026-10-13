# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261013_005113`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92000`

## run_t1

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261013_005113/run_t1.yaml`

```yaml
timestamp: '2026-10-13T00:51:15Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/.worktrees/tenferro-benchmark-gpu-issues/extern/tenferro-rs
  commit: 38c568487b87f91a5270c88becb558b65b44d0a9
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/.worktrees/tenferro-benchmark-gpu-issues/extern/tenferro-rs
blas:
  implementation: none
environment:
  hostname: primerose
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: 1c0734932d8c82676421dbd83332ed77c67507ee
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.97.1 (8bab26f4f 2026-07-14)
    BENCH_INSTANCE: null
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 3, coverage `full`, effort `standard`
- Expected 140, selected 140, executed 140, unsupported 0, failed 0, missing 0, noisy 1.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `alloc_zero_f64_host-zero-upload_32768el@t1`

Numerical checks: **140/140 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 23157.5 | 1167.9 | 4.0% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 7036.2 | 46.7 | 0.5% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9989.3 | 86.2 | 0.7% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 26153.0 | 202.3 | 0.6% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7021.0 | 42.5 | 0.4% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 59455.3 | 2146.2 | 2.0% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 33683.9 | 137.3 | 1.5% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13698.1 | 45.9 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12593.2 | 213.6 | 1.1% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 262886.2 | 1644.3 | 1.3% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 103206.4 | 155.0 | 0.1% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50113.2 | 32.8 | 0.1% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1268771.5 | 11473.0 | 1.9% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 754444.5 | 1130.1 | 0.1% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636465.8 | 168.8 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 21676995.0 | 56631.0 | 0.2% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 13188301.0 | 21950.5 | 0.1% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10014579.0 | 571.0 | 0.0% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 20125.2 | 66.6 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7388.7 | 23.6 | 0.4% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9051.5 | 63.8 | 1.0% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 64 | 28737.1 | 1939.8 | 3.6% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8205.0 | 53.6 | 0.6% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8910.2 | 43.9 | 1.6% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 34484.8 | 1816.4 | 4.6% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 13759.3 | 67.2 | 0.4% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11775.5 | 103.2 | 0.7% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 592917.1 | 1463.1 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 97263.7 | 150.0 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49635.3 | 287.8 | 0.5% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 9057191.0 | 435814.8 | 2.8% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 979270.8 | 2400.0 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 656048.8 | 51.4 | 0.0% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 148096291.0 | 3050352.5 | 1.6% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 16038423.0 | 606297.5 | 2.4% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10343808.0 | 2866.0 | 0.0% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 2048 | 3294.3 | 21.5 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 25281.7 | 258.9 | 1.5% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 4055.5 | 33.9 | 0.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 25164.7 | 68.4 | 0.3% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 19256.5 | 196.8 | 0.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 39747.2 | 20.4 | 0.3% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 512 | 6288.3 | 43.4 | 3.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 40077.4 | 242.4 | 1.5% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 20226.8 | 227.4 | 0.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 40216.9 | 346.7 | 0.6% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 512 | 7178.7 | 74.1 | 0.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 40747.3 | 3957.8 | 5.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 20066.8 | 93.5 | 0.3% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42113.6 | 68.1 | 0.4% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 60272.4 | 320.3 | 38.2% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42416.2 | 479.5 | 1.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 21216.2 | 258.8 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 43767.5 | 326.3 | 0.7% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1092856.5 | 5910.5 | 0.5% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 42375.9 | 527.2 | 1.0% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 7570288.0 | 8850.0 | 3.3% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 18996.2 | 425.7 | 1.7% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 6242450.0 | 23856.0 | 0.4% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 16706.0 | 123.6 | 0.7% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 7172107.0 | 98463.0 | 2.5% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 18897.9 | 378.4 | 3.5% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 6899439.0 | 38886.5 | 0.8% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 128 | 21021.5 | 301.1 | 5.0% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5688084.0 | 8870.5 | 0.1% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2491122.0 | 20015.0 | 0.6% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 107487414.0 | 568616.5 | 0.4% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35998760.0 | 49251.5 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17430874.0 | 28951.0 | 0.1% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14326994.0 | 11730.5 | 0.1% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 296277049.0 | 985018.5 | 0.4% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 228054589.0 | 117068.5 | 0.6% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 354046.5 | 660.7 | 0.2% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 389151.2 | 1265.6 | 0.2% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 489859.2 | 1648.8 | 0.3% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 12749.6 | 255.9 | 1.5% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 31228.1 | 256.2 | 1.2% | ok |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 32 | 96035.6 | 464.2 | 0.4% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 26124.6 | 215.3 | 1.1% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 4 | 607447.5 | 2932.5 | 0.4% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 24739.9 | 213.9 | 0.5% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 1 | 2241685.0 | 10080.0 | 0.3% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 26008.9 | 149.6 | 0.5% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 593732.2 | 731.4 | 0.1% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 593717.2 | 216.1 | 0.1% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4769978.0 | 1614.5 | 0.1% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4770108.0 | 2150.5 | 0.1% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6568751.0 | 17119.5 | 0.5% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3513392.0 | 6965.5 | 0.2% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41394689.0 | 14250.5 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41444010.0 | 8464.5 | 0.0% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32586144.0 | 8770.5 | 0.1% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9356111.0 | 12820.5 | 0.1% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 389078224.0 | 23429.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 388776359.0 | 21746.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 263114589.0 | 19080.0 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 62234775.0 | 21266.0 | 0.1% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 307908.9 | 1283.1 | 0.3% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 303837.5 | 885.5 | 0.3% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 210382.4 | 831.0 | 0.3% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 451888.0 | 1704.4 | 0.3% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 450964.2 | 2253.2 | 0.5% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 413907.0 | 2178.9 | 0.5% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 1 | 2014098.0 | 3996.0 | 0.3% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 2 | 1985178.0 | 4792.8 | 0.2% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19876027.0 | 11410.5 | 0.0% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 335349.6 | 10924.7 | 1.8% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 317884.2 | 1124.4 | 0.3% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 270249.1 | 369.4 | 0.2% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 489514.1 | 2671.9 | 0.7% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 485464.0 | 6732.1 | 0.9% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 668246.8 | 8505.2 | 1.0% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3311036.0 | 42131.0 | 1.0% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3320396.0 | 7840.5 | 0.3% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49723622.0 | 45181.0 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 54752.2 | 32.3 | 0.1% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 256 | 7838.5 | 155.4 | 2.9% | ok |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 512 | 8178.7 | 164.1 | 1.4% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 2 | 7100.0 | 620.0 | 6.8% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 33022985.0 | 4835.5 | 0.0% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 11107.9 | 56.1 | 0.5% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 33341.1 | 264.5 | 0.8% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 4 | 488574.0 | 2841.1 | 0.4% | ok |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 66270.7 | 498.8 | 0.6% | ok |
| vendor_handle_cublas_f64_runtime-handle | #2083 | runtime-handle | tenferro-cuda | f64 |  | 1 | 64 | 59032.5 | 205.4 | 0.3% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so a sample is exactly one call: the median is that call's latency, not an average over a batch, and the precondition sits outside its clock.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 94503.0 | 6425.5 | 7.0% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 43701.0 | 559.5 | 1.3% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 48972.0 | 1745.5 | 2.6% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 50992.0 | 2265.5 | 4.3% | ok |
| borrowed_input_gemm_f64_n8_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 42981.0 | 740.0 | 3.2% | ok |
| borrowed_input_gemm_f64_n8_view-input | #1925 | view-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 42341.0 | 1030.5 | 6.9% | ok |
| borrowed_input_gemm_f64_n32_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 43231.0 | 954.5 | 1.5% | ok |
| borrowed_input_gemm_f64_n32_view-input | #1925 | view-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 46211.0 | 726.0 | 1.3% | ok |

## State diagnostics (no timing)

Allocation and lifetime state, read before and after each arm: driver free device bytes for the pool, and process RSS where the issue's observable is host memory (#1924). Bytes, not nanoseconds; a case that keeps memory it no longer needs shows up as a non-zero retained or growth value, and a probe that the driver refuses shows up in `probe_status`.

### vendor_churn_f64_cycles8 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 8 | — |
| free_bytes_after | 84505591808 | 80590.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 666173440 | 635.3 |
| rss_bytes_before | 652103680 | 621.9 |
| rss_growth_bytes | 14069760 | 13.4 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 84505591808 | 652103680 |
| 1 | 0 | 84505591808 | 666165248 |
| 2 | 0 | 84505591808 | 666173440 |
| 3 | 0 | 84505591808 | 666173440 |
| 4 | 0 | 84505591808 | 666173440 |
| 5 | 0 | 84505591808 | 666173440 |
| 6 | 0 | 84505591808 | 666173440 |
| 7 | 0 | 84505591808 | 666173440 |

### vendor_churn_f64_cycles64 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 64 | — |
| free_bytes_after | 84505591808 | 80590.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 666263552 | 635.4 |
| rss_bytes_before | 652152832 | 621.9 |
| rss_growth_bytes | 14110720 | 13.5 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 84505591808 | 652152832 |
| 1 | 0 | 84505591808 | 666206208 |
| 2 | 0 | 84505591808 | 666214400 |
| 3 | 0 | 84505591808 | 666214400 |
| 4 | 0 | 84505591808 | 666214400 |
| 5 | 0 | 84505591808 | 666214400 |
| 6 | 0 | 84505591808 | 666214400 |
| 7 | 0 | 84505591808 | 666214400 |
| 8 | 0 | 84505591808 | 666214400 |
| 9 | 0 | 84505591808 | 666214400 |
| 10 | 0 | 84505591808 | 666214400 |
| 11 | 0 | 84505591808 | 666218496 |
| 12 | 0 | 84505591808 | 666218496 |
| 13 | 0 | 84505591808 | 666222592 |
| 14 | 0 | 84505591808 | 666222592 |
| 15 | 0 | 84505591808 | 666222592 |
| 16 | 0 | 84505591808 | 666222592 |
| 17 | 0 | 84505591808 | 666222592 |
| 18 | 0 | 84505591808 | 666226688 |
| 19 | 0 | 84505591808 | 666234880 |
| 20 | 0 | 84505591808 | 666234880 |
| 21 | 0 | 84505591808 | 666234880 |
| 22 | 0 | 84505591808 | 666234880 |
| 23 | 0 | 84505591808 | 666238976 |
| 24 | 0 | 84505591808 | 666238976 |
| 25 | 0 | 84505591808 | 666238976 |
| 26 | 0 | 84505591808 | 666238976 |
| 27 | 0 | 84505591808 | 666238976 |
| 28 | 0 | 84505591808 | 666238976 |
| 29 | 0 | 84505591808 | 666238976 |
| 30 | 0 | 84505591808 | 666238976 |
| 31 | 0 | 84505591808 | 666243072 |
| 32 | 0 | 84505591808 | 666243072 |
| 33 | 0 | 84505591808 | 666243072 |
| 34 | 0 | 84505591808 | 666243072 |
| 35 | 0 | 84505591808 | 666243072 |
| 36 | 0 | 84505591808 | 666243072 |
| 37 | 0 | 84505591808 | 666255360 |
| 38 | 0 | 84505591808 | 666255360 |
| 39 | 0 | 84505591808 | 666255360 |
| 40 | 0 | 84505591808 | 666255360 |
| 41 | 0 | 84505591808 | 666255360 |
| 42 | 0 | 84505591808 | 666259456 |
| 43 | 0 | 84505591808 | 666259456 |
| 44 | 0 | 84505591808 | 666259456 |
| 45 | 0 | 84505591808 | 666259456 |
| 46 | 0 | 84505591808 | 666259456 |
| 47 | 0 | 84505591808 | 666259456 |
| 48 | 0 | 84505591808 | 666259456 |
| 49 | 0 | 84505591808 | 666259456 |
| 50 | 0 | 84505591808 | 666259456 |
| 51 | 0 | 84505591808 | 666259456 |
| 52 | 0 | 84505591808 | 666259456 |
| 53 | 0 | 84505591808 | 666259456 |
| 54 | 0 | 84505591808 | 666259456 |
| 55 | 0 | 84505591808 | 666259456 |
| 56 | 0 | 84505591808 | 666259456 |
| 57 | 0 | 84505591808 | 666259456 |
| 58 | 0 | 84505591808 | 666259456 |
| 59 | 0 | 84505591808 | 666259456 |
| 60 | 0 | 84505591808 | 666259456 |
| 61 | 0 | 84505591808 | 666263552 |
| 62 | 0 | 84505591808 | 666263552 |
| 63 | 0 | 84505591808 | 666263552 |

### pool_retention_f64_alloc-free (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 76043583488 | 72520.8 |
| free_bytes_after_recreate | 76043583488 | 72520.8 |
| free_bytes_start | 84633518080 | 80712.8 |
| free_bytes_while_alive | 76043583488 | 72520.8 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (alloc-free arm)` | — |
| retained_bytes_after_drop | 8589934592 | 8192.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |

### pool_retention_f64_runtime-recreate (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 76043583488 | 72520.8 |
| free_bytes_after_recreate | 76043583488 | 72520.8 |
| free_bytes_start | 84633518080 | 80712.8 |
| free_bytes_while_alive | 76043583488 | 72520.8 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (runtime-recreate arm)` | — |
| retained_bytes_after_drop | 8589934592 | 8192.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |

### pool_retention_f64_driver-alloc-after-free (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 76043583488 | 72520.8 |
| free_bytes_after_recreate | 76043583488 | 72520.8 |
| free_bytes_start | 84633518080 | 80712.8 |
| free_bytes_while_alive | 76043583488 | 72520.8 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `ok (6442450944 bytes)` | — |
| retained_bytes_after_drop | 8589934592 | 8192.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |


## First-call diagnostics (not steady-state rows)

One cold process; each distinct layout is uploaded outside timing, then its first and second call are timed separately (each includes a device synchronize).

### first_call_transpose_to-contiguous_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.074 ms, second call 0.037 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 285.387 | 0.083 |
| [2, 5] -> transpose |  | 0.087 | 0.043 |
| [2, 7] -> transpose |  | 0.081 | 0.042 |
| [3, 5] -> transpose |  | 0.081 | 0.041 |
| [3, 7] -> transpose |  | 0.078 | 0.041 |
| [4, 5] -> transpose |  | 0.079 | 0.049 |
| [4, 7] -> transpose |  | 0.070 | 0.037 |
| [5, 5] -> transpose |  | 0.076 | 0.037 |
| [5, 7] -> transpose |  | 0.074 | 0.037 |
| [6, 5] -> transpose |  | 0.075 | 0.037 |
| [6, 7] -> transpose |  | 0.073 | 0.047 |
| [7, 5] -> transpose |  | 0.077 | 0.036 |
| [7, 7] -> transpose |  | 0.073 | 0.036 |
| [8, 5] -> transpose |  | 0.072 | 0.037 |
| [8, 7] -> transpose |  | 0.077 | 0.035 |
| [9, 5] -> transpose |  | 0.069 | 0.043 |
| [9, 7] -> transpose |  | 0.075 | 0.036 |
| [10, 5] -> transpose |  | 0.071 | 0.037 |
| [10, 7] -> transpose |  | 0.070 | 0.036 |
| [11, 5] -> transpose |  | 0.071 | 0.037 |
| [11, 7] -> transpose |  | 0.069 | 0.035 |
| [12, 5] -> transpose |  | 0.073 | 0.037 |
| [12, 7] -> transpose |  | 0.074 | 0.036 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.029 ms, second call 0.026 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 48.504 | 0.047 |
| [2, 5] -> transpose |  | 0.052 | 0.037 |
| [2, 7] -> transpose |  | 0.038 | 0.035 |
| [3, 5] -> transpose |  | 0.038 | 0.034 |
| [3, 7] -> transpose |  | 0.037 | 0.033 |
| [4, 5] -> transpose |  | 0.038 | 0.034 |
| [4, 7] -> transpose |  | 0.036 | 0.034 |
| [5, 5] -> transpose |  | 0.028 | 0.025 |
| [5, 7] -> transpose |  | 0.029 | 0.025 |
| [6, 5] -> transpose |  | 0.027 | 0.025 |
| [6, 7] -> transpose |  | 0.027 | 0.025 |
| [7, 5] -> transpose |  | 0.028 | 0.026 |
| [7, 7] -> transpose |  | 0.032 | 0.026 |
| [8, 5] -> transpose |  | 0.027 | 0.026 |
| [8, 7] -> transpose |  | 0.029 | 0.025 |
| [9, 5] -> transpose |  | 0.028 | 0.025 |
| [9, 7] -> transpose |  | 0.028 | 0.025 |
| [10, 5] -> transpose |  | 0.026 | 0.025 |
| [10, 7] -> transpose |  | 0.029 | 0.025 |
| [11, 5] -> transpose |  | 0.027 | 0.025 |
| [11, 7] -> transpose |  | 0.027 | 0.026 |
| [12, 5] -> transpose |  | 0.032 | 0.026 |
| [12, 7] -> transpose |  | 0.028 | 0.025 |


## Timing boundaries

- **alloc_zero / alloc-fill-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_output + fill_zero_write (device fill), device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_zero_output, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: one with_cubecl entry/exit per batch, alloc_zero_output x iterations, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **borrowed_input_gemm / owned-input** — inside: one small dot_general_read with owned inputs, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **borrowed_input_gemm / view-input** — inside: one small dot_general_read with a borrowed transposed input view, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **contraction_plan_churn / distinct-plans** — inside: one re-run of the first contraction after the plan cache was filled by distinct shapes, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / repeated-plan** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **eager_op / eager-per-call** — inside: one eager op with a with_eager_session entry/exit per operation, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session** — inside: one eager op in a reused with_eager_session, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevd** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevd** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevj** — inside: one batched eigh call ([n, n, batch], batch trailing), device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevj** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **first_call_layouts / copy-into** — inside: to_contiguous (or copy_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **first_call_layouts / to-contiguous** — inside: to_contiguous (or copy_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **gather_table / cudarc-row-copies** — inside: 8 device-to-device row copies (cudarc) into a freshly allocated destination, stream.synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **gather_table / eager-gather** — inside: one eager gather of 8 indices from a [vocab, hidden] f32 device table, one device synchronize at the batch edge; outside: backend/runtime/context construction, table generation and upload, correctness check (download).
- **hadamard_dot_general / dot-general** — inside: one dot_general with every axis as a batch axis (pure Hadamard), device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **hadamard_dot_general / multiply** — inside: one elementwise multiply, device synchronize at the batch edge; outside: backend construction, operand upload, correctness check (download).
- **owned_destination / borrowed-dst** — inside: dot_general_read_into a reused borrowed-view destination, one isolated call per sample; device synchronize after the call; outside: backend construction, operand and destination upload, the destination handle built per call, correctness check (download).
- **owned_destination / owned-dst** — inside: dot_general_read_into a reused owned destination, one isolated call per sample; device synchronize after the call; outside: backend construction, operand and destination upload, the destination handle built per call, correctness check (download).
- **pool_retention / alloc-free** — inside: —; outside: everything (a state diagnostic, no timing).
- **pool_retention / driver-alloc-after-free** — inside: —; outside: everything (a state diagnostic, no timing).
- **pool_retention / runtime-recreate** — inside: —; outside: everything (a state diagnostic, no timing).
- **raw_scope_entry / empty-behind-queued-gemm** — inside: the second empty with_raw scope after queued device work; outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **raw_scope_entry / empty-idle** — inside: with_cuda_exec_session + empty with_raw closure on an idle device; outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **raw_scope_entry / sync-behind-queued-gemm** — inside: explicit device synchronize behind queued device work (the reference barrier); outside: backend and runtime construction, operand upload, queued GEMM submissions and the first empty scope (the precondition of the timed scope), correctness probe (download).
- **reshape_materialize / constant** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
- **reshape_materialize / op-output** — inside: one reshape of a [2^26] f32 device tensor to [2^16, 2^10], one device synchronize at the batch edge; outside: backend/runtime construction, host fill, upload or elementwise production of the input, correctness check (download).
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
- **vendor_churn / backend-cycle** — inside: —; outside: everything (a state diagnostic, no timing).
- **vendor_handle / create-destroy** — inside: cublasCreate + cublasSetStream + dgemm + cublasDestroy inside one with_raw scope per operation, host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.
- **vendor_handle / reused** — inside: cublasSetStream + dgemm inside one with_raw scope per operation (handle reused), host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.
- **vendor_handle / runtime-handle** — inside: dgemm inside one with_raw scope per operation, on the runtime's own cuBLAS handle, host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.

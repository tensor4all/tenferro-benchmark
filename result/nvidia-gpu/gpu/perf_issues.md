# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261011_233315`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261011_233315/run_t1.yaml`

```yaml
timestamp: '2026-10-11T23:33:17Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/.worktrees/tenferro-benchmark-gpu-issues/extern/tenferro-rs
  commit: a3b40ff0b5af0932c2bbf959d7bd2326663b5e9e
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
    BENCHMARK_COMMIT: 57ffe6c3c62f59ca9a89ea4554e1ccf23903873c
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.97.1 (8bab26f4f 2026-07-14)
    BENCH_INSTANCE: null
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 2, coverage `full`, effort `standard`
- Expected 135, selected 135, executed 135, unsupported 0, failed 0, missing 0, noisy 3.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `transfer_up_f64_tenferro_268435456B@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`, `qr_live_buffers_f64_n32_live16@t1`

Numerical checks: **135/135 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 21989.2 | 70.6 | 0.6% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6566.7 | 287.0 | 2.8% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 10450.0 | 115.8 | 1.4% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 24406.7 | 637.1 | 3.1% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7019.6 | 19.7 | 0.4% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 58990.8 | 506.4 | 0.9% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 34274.6 | 689.1 | 2.2% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13653.9 | 657.5 | 4.0% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12528.9 | 124.5 | 0.8% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 254839.0 | 781.8 | 0.2% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 89213.3 | 4070.8 | 6.1% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 49749.6 | 180.4 | 0.3% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1259318.0 | 39791.2 | 2.3% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 748095.2 | 671.4 | 0.2% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636531.5 | 349.9 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 22292292.0 | 2639434.5 | 10.1% | NOISY |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 12936470.0 | 45557.0 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10014552.0 | 5415.0 | 0.8% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 20833.0 | 81.0 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7606.1 | 101.3 | 0.9% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9433.7 | 238.2 | 1.9% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 64 | 28428.0 | 170.3 | 0.4% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8310.5 | 30.6 | 0.3% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9507.6 | 194.9 | 2.0% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 30875.8 | 2421.3 | 5.0% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 13878.1 | 288.8 | 1.4% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11674.2 | 119.9 | 2.6% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 4 | 603843.2 | 2013.8 | 0.5% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 95726.9 | 2085.7 | 1.8% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49578.7 | 291.4 | 0.5% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 9077984.0 | 290676.2 | 2.3% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 1137509.0 | 3055.2 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 656407.5 | 193.6 | 0.0% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 149113547.0 | 1784709.0 | 1.2% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 14769537.0 | 23131.5 | 0.3% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10316602.0 | 2445.0 | 0.1% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 2048 | 3202.7 | 19.8 | 1.4% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 23965.2 | 97.5 | 3.6% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 3969.0 | 71.3 | 1.2% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 26699.6 | 914.3 | 2.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 19657.3 | 679.6 | 4.4% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 37169.6 | 727.0 | 3.6% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 512 | 6601.0 | 441.0 | 4.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 37468.0 | 124.5 | 0.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 18685.3 | 483.0 | 2.1% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 41460.8 | 2933.1 | 4.7% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 512 | 7260.6 | 440.6 | 3.6% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 41063.9 | 1606.4 | 3.6% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 21545.1 | 176.4 | 0.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 41908.3 | 221.0 | 2.3% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 59758.7 | 13180.4 | 35.4% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42917.4 | 1656.8 | 2.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 21356.4 | 211.6 | 0.8% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 44092.3 | 220.5 | 0.6% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1099943.0 | 50903.8 | 2.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 42112.5 | 328.6 | 0.9% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 6361223.0 | 39916.5 | 1.8% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 18367.5 | 205.7 | 0.9% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 6028863.0 | 23940.5 | 0.3% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 19445.2 | 2432.5 | 7.5% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 6883048.0 | 262132.5 | 3.4% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 18909.6 | 334.7 | 1.6% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 6843928.0 | 373716.5 | 3.8% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 128 | 22971.9 | 2149.4 | 5.3% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5723603.0 | 62716.5 | 1.2% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2502136.0 | 13830.5 | 0.6% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 107876773.0 | 1817364.5 | 1.4% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 36019743.0 | 56242.0 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17472011.0 | 26065.5 | 0.2% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14307655.0 | 70287.5 | 0.5% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 299583516.0 | 733000.5 | 0.2% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 226378839.0 | 153789.5 | 0.6% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 470403.0 | 62108.8 | 12.7% | NOISY |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 397088.4 | 631.9 | 0.2% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 486313.6 | 1483.1 | 0.3% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 11339.8 | 180.2 | 1.1% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 33794.6 | 1388.9 | 4.7% | ok |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 32 | 96473.2 | 385.2 | 0.3% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 25299.5 | 240.7 | 0.6% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 4 | 611971.2 | 3730.1 | 0.6% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 28218.8 | 470.1 | 1.8% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 1 | 2263299.0 | 10365.0 | 0.5% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 25590.1 | 280.0 | 1.0% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 595528.2 | 1593.9 | 0.8% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 594143.0 | 190.0 | 0.2% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4782247.0 | 9765.5 | 0.4% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4766325.0 | 5644.5 | 0.1% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6605672.0 | 41391.5 | 0.4% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3565989.0 | 13210.0 | 0.3% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41364075.0 | 8275.5 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41344835.0 | 13756.0 | 0.0% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32411111.0 | 14665.0 | 0.0% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9372687.0 | 16320.5 | 0.2% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 389364024.0 | 171326.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 388859935.0 | 27652.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 263029377.0 | 65007.0 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 62156167.0 | 21005.5 | 0.0% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 308960.8 | 1895.6 | 0.5% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 306047.0 | 1453.2 | 1.5% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 216481.7 | 640.3 | 1.4% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 458831.6 | 2392.4 | 1.3% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 464774.2 | 4758.9 | 1.5% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 426645.6 | 3855.1 | 1.8% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 1 | 2005272.0 | 11230.0 | 0.6% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 1 | 2009961.0 | 3900.5 | 0.1% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19965763.0 | 10065.0 | 0.1% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 328865.1 | 1391.9 | 0.3% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 321029.9 | 805.7 | 0.2% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 275729.8 | 3702.6 | 1.8% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 493638.9 | 2494.4 | 0.4% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 499969.1 | 8231.4 | 1.6% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 670905.8 | 1461.5 | 0.2% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3326332.0 | 28206.0 | 0.8% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3339393.0 | 14894.5 | 0.8% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49864662.0 | 26181.0 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 54701.4 | 28.0 | 0.1% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 256 | 7805.8 | 178.0 | 4.1% | ok |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 256 | 14661.6 | 56.8 | 1.5% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 32961213.0 | 196376.0 | 0.6% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 33019354.0 | 430493.0 | 0.9% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 11481.5 | 627.5 | 3.7% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 32158.3 | 271.7 | 1.4% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 4 | 524038.5 | 2262.5 | 0.5% | ok |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 74097.9 | 382.0 | 0.4% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so a sample is exactly one call: the median is that call's latency, not an average over a batch, and the precondition sits outside its clock.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 85572.0 | 360.5 | 0.9% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 40112.0 | 294.5 | 1.1% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 52531.0 | 550.5 | 1.0% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 52482.0 | 1490.5 | 2.0% | ok |

## State diagnostics (no timing)

Allocation and lifetime state of the CUDA pool, read from the driver before and after each arm. Bytes, not nanoseconds; a case that keeps memory it no longer needs shows up as a non-zero retained value, and a probe that the driver refuses shows up in `probe_status`.

### vendor_churn_f64_cycles8 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 8 | — |
| free_bytes_after | 84505591808 | 80590.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 0 | 0.0 |

| Cycle | Free bytes | Delta bytes |
|---:|---:|---:|
| 0 | 84505591808 | 0 |
| 1 | 84505591808 | 0 |
| 2 | 84505591808 | 0 |
| 3 | 84505591808 | 0 |
| 4 | 84505591808 | 0 |
| 5 | 84505591808 | 0 |
| 6 | 84505591808 | 0 |
| 7 | 84505591808 | 0 |

### vendor_churn_f64_cycles64 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 64 | — |
| free_bytes_after | 84505591808 | 80590.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 0 | 0.0 |

| Cycle | Free bytes | Delta bytes |
|---:|---:|---:|
| 0 | 84505591808 | 0 |
| 1 | 84505591808 | 0 |
| 2 | 84505591808 | 0 |
| 3 | 84505591808 | 0 |
| 4 | 84505591808 | 0 |
| 5 | 84505591808 | 0 |
| 6 | 84505591808 | 0 |
| 7 | 84505591808 | 0 |
| 8 | 84505591808 | 0 |
| 9 | 84505591808 | 0 |
| 10 | 84505591808 | 0 |
| 11 | 84505591808 | 0 |
| 12 | 84505591808 | 0 |
| 13 | 84505591808 | 0 |
| 14 | 84505591808 | 0 |
| 15 | 84505591808 | 0 |
| 16 | 84505591808 | 0 |
| 17 | 84505591808 | 0 |
| 18 | 84505591808 | 0 |
| 19 | 84505591808 | 0 |
| 20 | 84505591808 | 0 |
| 21 | 84505591808 | 0 |
| 22 | 84505591808 | 0 |
| 23 | 84505591808 | 0 |
| 24 | 84505591808 | 0 |
| 25 | 84505591808 | 0 |
| 26 | 84505591808 | 0 |
| 27 | 84505591808 | 0 |
| 28 | 84505591808 | 0 |
| 29 | 84505591808 | 0 |
| 30 | 84505591808 | 0 |
| 31 | 84505591808 | 0 |
| 32 | 84505591808 | 0 |
| 33 | 84505591808 | 0 |
| 34 | 84505591808 | 0 |
| 35 | 84505591808 | 0 |
| 36 | 84505591808 | 0 |
| 37 | 84505591808 | 0 |
| 38 | 84505591808 | 0 |
| 39 | 84505591808 | 0 |
| 40 | 84505591808 | 0 |
| 41 | 84505591808 | 0 |
| 42 | 84505591808 | 0 |
| 43 | 84505591808 | 0 |
| 44 | 84505591808 | 0 |
| 45 | 84505591808 | 0 |
| 46 | 84505591808 | 0 |
| 47 | 84505591808 | 0 |
| 48 | 84505591808 | 0 |
| 49 | 84505591808 | 0 |
| 50 | 84505591808 | 0 |
| 51 | 84505591808 | 0 |
| 52 | 84505591808 | 0 |
| 53 | 84505591808 | 0 |
| 54 | 84505591808 | 0 |
| 55 | 84505591808 | 0 |
| 56 | 84505591808 | 0 |
| 57 | 84505591808 | 0 |
| 58 | 84505591808 | 0 |
| 59 | 84505591808 | 0 |
| 60 | 84505591808 | 0 |
| 61 | 84505591808 | 0 |
| 62 | 84505591808 | 0 |
| 63 | 84505591808 | 0 |

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

Median over 22 distinct layouts: first call 0.076 ms, second call 0.037 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 292.214 | 0.085 |
| [2, 5] -> transpose |  | 0.087 | 0.045 |
| [2, 7] -> transpose |  | 0.085 | 0.043 |
| [3, 5] -> transpose |  | 0.084 | 0.044 |
| [3, 7] -> transpose |  | 0.081 | 0.037 |
| [4, 5] -> transpose |  | 0.073 | 0.038 |
| [4, 7] -> transpose |  | 0.081 | 0.038 |
| [5, 5] -> transpose |  | 0.076 | 0.037 |
| [5, 7] -> transpose |  | 0.075 | 0.037 |
| [6, 5] -> transpose |  | 0.074 | 0.037 |
| [6, 7] -> transpose |  | 0.072 | 0.037 |
| [7, 5] -> transpose |  | 0.078 | 0.038 |
| [7, 7] -> transpose |  | 0.076 | 0.036 |
| [8, 5] -> transpose |  | 0.073 | 0.037 |
| [8, 7] -> transpose |  | 0.079 | 0.037 |
| [9, 5] -> transpose |  | 0.069 | 0.037 |
| [9, 7] -> transpose |  | 0.080 | 0.040 |
| [10, 5] -> transpose |  | 0.081 | 0.038 |
| [10, 7] -> transpose |  | 0.071 | 0.037 |
| [11, 5] -> transpose |  | 0.070 | 0.037 |
| [11, 7] -> transpose |  | 0.069 | 0.037 |
| [12, 5] -> transpose |  | 0.078 | 0.039 |
| [12, 7] -> transpose |  | 0.073 | 0.038 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.042 ms, second call 0.036 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 48.400 | 0.043 |
| [2, 5] -> transpose |  | 0.038 | 0.031 |
| [2, 7] -> transpose |  | 0.034 | 0.030 |
| [3, 5] -> transpose |  | 0.032 | 0.028 |
| [3, 7] -> transpose |  | 0.034 | 0.030 |
| [4, 5] -> transpose |  | 0.033 | 0.032 |
| [4, 7] -> transpose |  | 0.040 | 0.036 |
| [5, 5] -> transpose |  | 0.041 | 0.036 |
| [5, 7] -> transpose |  | 0.044 | 0.036 |
| [6, 5] -> transpose |  | 0.042 | 0.035 |
| [6, 7] -> transpose |  | 0.043 | 0.037 |
| [7, 5] -> transpose |  | 0.041 | 0.036 |
| [7, 7] -> transpose |  | 0.043 | 0.044 |
| [8, 5] -> transpose |  | 0.043 | 0.037 |
| [8, 7] -> transpose |  | 0.049 | 0.038 |
| [9, 5] -> transpose |  | 0.042 | 0.037 |
| [9, 7] -> transpose |  | 0.042 | 0.036 |
| [10, 5] -> transpose |  | 0.041 | 0.047 |
| [10, 7] -> transpose |  | 0.043 | 0.038 |
| [11, 5] -> transpose |  | 0.042 | 0.036 |
| [11, 7] -> transpose |  | 0.043 | 0.035 |
| [12, 5] -> transpose |  | 0.027 | 0.023 |
| [12, 7] -> transpose |  | 0.028 | 0.023 |


## Timing boundaries

- **alloc_zero / alloc-fill-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_output + fill_zero_write (device fill), device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_zero_output, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: one with_cubecl entry/exit per batch, alloc_zero_output x iterations, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **contraction_plan_churn / distinct-plans** — inside: one re-run of the first contraction after the plan cache was filled by distinct shapes, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / repeated-plan** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **eager_op / eager-per-call** — inside: one eager op with a with_eager_session entry/exit per operation, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session** — inside: one eager op in a reused with_eager_session, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eigh_driver / auto** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
- **eigh_driver / syevd** — inside: one batched eigh call ([n, n, batch], batch trailing), c64 input, device synchronize at the batch edge; outside: backend construction, host batch construction and upload, residual check (download).
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
- **vendor_churn / backend-cycle** — inside: —; outside: everything (a state diagnostic, no timing).
- **vendor_handle / create-destroy** — inside: cublasCreate + cublasSetStream + dgemm + cublasDestroy inside one with_raw scope per operation, host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.
- **vendor_handle / reused** — inside: cublasSetStream + dgemm inside one with_raw scope per operation (handle reused), host barrier (raw.synchronize) inside the scope; outside: backend construction, operand upload, correctness probe, persistent handle creation for the reused arm.

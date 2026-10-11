# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261011_232352`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261011_232352/run_t1.yaml`

```yaml
timestamp: '2026-10-11T23:23:54Z'
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
    BENCHMARK_COMMIT: 6b80b18cba0fe696fe814c61ed2ecb7beb76b78f
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.97.1 (8bab26f4f 2026-07-14)
    BENCH_INSTANCE: null
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 2, coverage `full`, effort `standard`
- Expected 135, selected 135, executed 135, unsupported 0, failed 0, missing 0, noisy 2.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `alloc_zero_f64_alloc-zero-per-callback_0el@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`

Numerical checks: **135/135 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 20658.5 | 250.6 | 3.6% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6245.5 | 222.5 | 2.2% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9976.8 | 131.9 | 1.2% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 23568.8 | 285.3 | 0.7% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7470.3 | 264.5 | 2.2% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 32 | 65891.3 | 366.1 | 1.4% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 34321.9 | 118.9 | 0.2% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13687.2 | 51.9 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12553.2 | 21.3 | 0.6% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 271702.9 | 3107.0 | 1.4% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 89176.2 | 192.6 | 0.2% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50053.9 | 68.3 | 0.2% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1245240.0 | 3200.0 | 0.4% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 772828.8 | 15352.9 | 1.3% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636470.5 | 133.9 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 21870101.0 | 107122.0 | 1.2% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 13365465.0 | 60996.5 | 0.8% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10014874.0 | 1155.5 | 0.8% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 20779.6 | 1746.8 | 5.5% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7437.2 | 299.7 | 2.7% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9297.3 | 36.0 | 0.8% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 27866.5 | 632.9 | 2.4% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8103.6 | 442.1 | 3.2% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9885.2 | 244.3 | 2.1% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 34327.2 | 185.9 | 0.7% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 13765.2 | 88.0 | 0.5% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11929.1 | 100.6 | 0.8% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 611300.0 | 3203.1 | 3.8% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 95648.3 | 2724.5 | 1.8% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49994.2 | 55.3 | 0.1% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 9324168.5 | 397762.5 | 2.6% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 972203.5 | 80559.5 | 4.4% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 654128.5 | 118.8 | 0.1% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 145691571.0 | 5565960.0 | 3.2% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 15529761.0 | 164609.5 | 1.2% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10332754.0 | 86152.5 | 1.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 2048 | 3781.1 | 15.7 | 5.3% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 22962.3 | 1626.9 | 11.1% | NOISY |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 512 | 3987.7 | 64.9 | 2.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 25068.0 | 789.9 | 2.5% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 19810.9 | 234.5 | 0.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 43365.0 | 3216.3 | 4.1% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 512 | 7040.0 | 64.3 | 0.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 37272.3 | 143.6 | 0.3% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 19944.3 | 1250.7 | 5.4% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 36943.4 | 315.6 | 2.9% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 512 | 6727.0 | 48.0 | 0.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 40144.7 | 212.7 | 0.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 21357.9 | 1707.3 | 4.5% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 41778.5 | 3255.5 | 4.7% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 61440.2 | 410.0 | 32.6% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 43965.1 | 3903.9 | 5.8% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 20157.7 | 1678.3 | 5.3% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 41378.2 | 2285.7 | 3.7% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1088762.0 | 3262.2 | 0.3% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 42374.2 | 334.7 | 0.5% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 6761067.0 | 121452.5 | 1.6% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 20987.5 | 1678.0 | 5.9% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 6480031.0 | 306028.5 | 3.1% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 17899.4 | 1083.8 | 4.1% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 7476856.0 | 453036.5 | 3.5% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 20164.3 | 1893.8 | 5.3% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 6790359.0 | 205440.5 | 2.7% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 64 | 21762.6 | 1013.8 | 3.2% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5512217.0 | 272331.5 | 2.8% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2474881.0 | 7104.5 | 0.2% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 106828197.0 | 1301738.0 | 0.9% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 36009096.0 | 67386.5 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17456834.0 | 85647.5 | 0.6% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14298336.0 | 133113.0 | 0.8% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 301528139.0 | 6317726.5 | 1.2% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 227877317.0 | 753389.5 | 0.5% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 340813.5 | 5692.1 | 2.3% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 386713.4 | 26541.9 | 8.6% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 490653.5 | 1420.7 | 0.3% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 11386.5 | 358.2 | 4.7% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 35998.4 | 3009.2 | 5.5% | ok |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 32 | 96719.3 | 326.2 | 0.5% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 24645.2 | 158.3 | 2.6% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 4 | 610272.8 | 2091.1 | 0.3% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 24864.5 | 248.7 | 0.6% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 1 | 2264697.0 | 10199.5 | 0.3% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 26071.4 | 584.6 | 1.4% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 594672.5 | 995.0 | 0.1% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 593220.2 | 592.5 | 0.1% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4764212.0 | 12955.0 | 0.2% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4721551.0 | 10885.5 | 0.2% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6532577.0 | 26845.5 | 0.4% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3559741.0 | 9715.5 | 0.3% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41399481.0 | 64571.0 | 0.2% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41193466.0 | 35940.5 | 0.1% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32326819.0 | 55951.5 | 0.2% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9357080.0 | 15785.5 | 0.2% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 389169680.0 | 101071.0 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 389190414.0 | 35695.0 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 263149883.0 | 51530.0 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 62237203.0 | 51256.0 | 0.1% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 285422.4 | 1015.1 | 0.4% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 304262.9 | 10204.6 | 2.3% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 215679.4 | 621.5 | 0.2% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 415723.2 | 3370.7 | 0.6% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 421385.9 | 4630.2 | 0.9% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 427747.4 | 6173.2 | 1.3% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 2 | 1953636.0 | 4482.2 | 0.2% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 1 | 2025133.0 | 6765.5 | 0.6% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19986260.0 | 164959.0 | 0.6% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 330663.6 | 11262.2 | 1.9% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 334017.4 | 1286.4 | 0.4% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 270802.0 | 900.7 | 0.3% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 493382.9 | 3470.8 | 1.2% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 487452.6 | 1484.4 | 2.5% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 663242.2 | 5662.6 | 1.0% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3337067.0 | 48021.5 | 4.8% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3321587.0 | 52210.5 | 1.0% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49757977.0 | 77207.0 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 54651.1 | 53.3 | 0.1% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 256 | 9018.9 | 522.3 | 4.2% | ok |

## Per-call session entry diagnostics (not operation comparisons)

A session (or eager session) is entered and left for every operation, inside the timer.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 256 | 14932.9 | 478.6 | 2.6% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 33001308.0 | 31081.0 | 0.7% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 33023079.0 | 31821.0 | 0.5% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 12041.1 | 730.3 | 4.0% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 31726.4 | 191.7 | 0.5% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 4 | 488280.0 | 5568.9 | 0.7% | ok |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 75186.9 | 843.3 | 1.6% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so each sample is one isolated call with its own clock, and its precondition is outside it.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 32 | 86994.5 | 929.3 | 2.1% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 64 | 38325.7 | 780.1 | 1.4% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 64 | 48695.2 | 153.9 | 0.3% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 64 | 51226.2 | 167.6 | 0.3% | ok |

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

Median over 22 distinct layouts: first call 0.072 ms, second call 0.035 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 297.934 | 0.079 |
| [2, 5] -> transpose |  | 0.086 | 0.042 |
| [2, 7] -> transpose |  | 0.076 | 0.041 |
| [3, 5] -> transpose |  | 0.080 | 0.039 |
| [3, 7] -> transpose |  | 0.077 | 0.041 |
| [4, 5] -> transpose |  | 0.083 | 0.040 |
| [4, 7] -> transpose |  | 0.069 | 0.034 |
| [5, 5] -> transpose |  | 0.072 | 0.035 |
| [5, 7] -> transpose |  | 0.074 | 0.035 |
| [6, 5] -> transpose |  | 0.073 | 0.035 |
| [6, 7] -> transpose |  | 0.069 | 0.034 |
| [7, 5] -> transpose |  | 0.074 | 0.035 |
| [7, 7] -> transpose |  | 0.067 | 0.033 |
| [8, 5] -> transpose |  | 0.069 | 0.034 |
| [8, 7] -> transpose |  | 0.073 | 0.033 |
| [9, 5] -> transpose |  | 0.066 | 0.032 |
| [9, 7] -> transpose |  | 0.076 | 0.036 |
| [10, 5] -> transpose |  | 0.068 | 0.034 |
| [10, 7] -> transpose |  | 0.068 | 0.033 |
| [11, 5] -> transpose |  | 0.066 | 0.033 |
| [11, 7] -> transpose |  | 0.065 | 0.033 |
| [12, 5] -> transpose |  | 0.070 | 0.038 |
| [12, 7] -> transpose |  | 0.071 | 0.035 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.038 ms, second call 0.033 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 46.198 | 0.055 |
| [2, 5] -> transpose |  | 0.045 | 0.037 |
| [2, 7] -> transpose |  | 0.039 | 0.033 |
| [3, 5] -> transpose |  | 0.039 | 0.033 |
| [3, 7] -> transpose |  | 0.049 | 0.036 |
| [4, 5] -> transpose |  | 0.040 | 0.036 |
| [4, 7] -> transpose |  | 0.039 | 0.035 |
| [5, 5] -> transpose |  | 0.039 | 0.035 |
| [5, 7] -> transpose |  | 0.050 | 0.042 |
| [6, 5] -> transpose |  | 0.038 | 0.034 |
| [6, 7] -> transpose |  | 0.038 | 0.034 |
| [7, 5] -> transpose |  | 0.039 | 0.034 |
| [7, 7] -> transpose |  | 0.041 | 0.036 |
| [8, 5] -> transpose |  | 0.032 | 0.024 |
| [8, 7] -> transpose |  | 0.030 | 0.025 |
| [9, 5] -> transpose |  | 0.031 | 0.030 |
| [9, 7] -> transpose |  | 0.029 | 0.026 |
| [10, 5] -> transpose |  | 0.029 | 0.025 |
| [10, 7] -> transpose |  | 0.027 | 0.026 |
| [11, 5] -> transpose |  | 0.028 | 0.025 |
| [11, 7] -> transpose |  | 0.030 | 0.026 |
| [12, 5] -> transpose |  | 0.029 | 0.026 |
| [12, 7] -> transpose |  | 0.027 | 0.026 |


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
- **owned_destination / borrowed-dst** — inside: dot_general_read_into a reused borrowed-view destination, device synchronize after each call; outside: backend construction, operand and destination upload, the destination handle the rule keeps out of a batched interval, correctness check (download).
- **owned_destination / owned-dst** — inside: dot_general_read_into a reused owned destination, device synchronize after each call; outside: backend construction, operand and destination upload, the destination handle the rule keeps out of a batched interval, correctness check (download).
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

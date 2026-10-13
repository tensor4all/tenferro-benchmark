# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261013_040229`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261013_040229/run_t1.yaml`

```yaml
timestamp: '2026-10-13T04:02:31Z'
tenferro_rs:
  path: /home/shinaoka/tensor4all/.worktrees/tenferro-benchmark-gpu-issues/extern/tenferro-rs
  commit: e1e0bd78d68e53bef533aff215c2383b58302003
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
    BENCHMARK_COMMIT: fb0cb4d62508ba3865b812b2f7437fa601d1a886
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.97.1 (8bab26f4f 2026-07-14)
    BENCH_INSTANCE: null
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 3, coverage `full`, effort `standard`
- Expected 146, selected 146, executed 145, unsupported 0, failed 1, missing 0, noisy 12.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Failed: `eager_op_f32_dot_general_decode_launch-only-floor@t1`
- Noisy: `transfer_up_f64_tenferro_8B@t1`, `transfer_up_f64_tenferro_4096B@t1`, `transfer_up_f64_cudarc-pageable_1048576B@t1`, `transfer_up_f64_cudarc-pageable_16777216B@t1`, `transfer_down_f64_tenferro_1048576B@t1`, `transfer_down_f64_cudarc-pageable_16777216B@t1`, `transfer_down_f64_cudarc-pinned_16777216B@t1`, `alloc_zero_f64_host-zero-upload_16el@t1`, `alloc_zero_f64_host-zero-upload_512el@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`, `hadamard_f32_1024x1024_multiply@t1`, `borrowed_input_gemm_f64_n8_owned-input@t1`

Numerical checks: **145/146 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 31712.7 | 6521.4 | 15.3% | NOISY |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6939.8 | 20.2 | 0.5% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9890.3 | 89.1 | 1.0% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 31838.3 | 4835.4 | 10.7% | NOISY |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 256 | 7625.1 | 306.5 | 3.4% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 55802.2 | 150.9 | 0.2% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 40338.9 | 2067.6 | 3.7% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 14152.8 | 84.4 | 0.5% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13654.6 | 1456.5 | 6.5% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 252374.9 | 2149.5 | 0.6% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 16 | 89967.4 | 695.7 | 17.1% | NOISY |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50202.6 | 56.1 | 0.1% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1408207.5 | 31871.2 | 1.7% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 867399.2 | 58059.4 | 10.7% | NOISY |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 637124.0 | 101.1 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 27746525.0 | 2147737.0 | 8.4% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 17140148.0 | 485571.0 | 2.5% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10020747.0 | 11981.5 | 2.3% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 21207.1 | 248.2 | 1.1% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 7815.8 | 29.7 | 0.3% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9442.1 | 199.4 | 1.9% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 29221.6 | 482.0 | 1.3% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8576.5 | 61.4 | 0.5% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9731.6 | 132.1 | 1.2% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 35110.7 | 154.4 | 0.3% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 15311.9 | 946.4 | 6.3% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11847.6 | 118.4 | 0.7% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 724261.9 | 74980.6 | 13.5% | NOISY |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 106661.1 | 343.2 | 0.9% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49989.8 | 45.1 | 0.2% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 10985230.0 | 713946.5 | 4.9% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 2 | 2622268.0 | 1210420.5 | 28.8% | NOISY |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 655724.5 | 135079.5 | 34.6% | NOISY |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 173905024.0 | 9256842.0 | 4.3% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 18881745.0 | 340287.0 | 1.4% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10370409.0 | 189506.5 | 1.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 2375.8 | 42.6 | 2.5% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 25175.9 | 124.8 | 0.5% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 512 | 4056.9 | 156.2 | 2.9% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 26329.6 | 1005.3 | 2.6% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 17726.9 | 304.7 | 1.3% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 40129.0 | 382.2 | 0.7% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 256 | 21301.1 | 8549.1 | 29.8% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 39023.0 | 134.6 | 0.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 18744.6 | 175.8 | 0.8% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 42725.6 | 3948.9 | 5.1% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 256 | 22178.1 | 8965.9 | 29.5% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 38835.8 | 321.3 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 18527.2 | 119.9 | 0.5% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42378.0 | 187.2 | 0.4% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 64057.2 | 4092.3 | 31.6% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42018.9 | 1008.2 | 1.5% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 22185.6 | 646.2 | 1.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 46310.0 | 866.8 | 1.6% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1178224.5 | 3415.2 | 0.3% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 44946.8 | 749.9 | 1.8% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 3794658.0 | 23970.5 | 0.5% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 15673.6 | 83.0 | 0.7% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 3637373.0 | 162430.5 | 3.4% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 16021.5 | 97.6 | 0.6% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 4653357.0 | 23301.0 | 0.5% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 22254.6 | 1094.0 | 2.9% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 5268147.0 | 30146.0 | 2.1% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 64 | 21613.1 | 186.9 | 0.7% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 7493422.0 | 376833.5 | 8.1% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2486883.0 | 2176.0 | 0.2% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 108255418.0 | 476285.5 | 0.5% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35997670.0 | 49672.0 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17507538.0 | 77618.5 | 0.7% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14419514.0 | 59637.0 | 0.3% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 298906520.0 | 872814.5 | 0.2% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 229534548.0 | 310911.5 | 0.7% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 357555.9 | 4950.8 | 1.0% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 374345.0 | 6998.3 | 2.8% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 416009.0 | 1053.1 | 3.1% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 10364.9 | 400.4 | 4.1% | ok |
| eager_op_f32_add_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 10585.3 | 209.4 | 1.3% | ok |
| eager_op_f32_add_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 512 | 5895.2 | 158.1 | 3.7% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 29677.1 | 39.2 | 0.3% | ok |
| eager_op_f32_dot_general_decode_eager-session-streaming | #2079 | eager-session-streaming | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 30655.4 | 526.7 | 1.6% | ok |
| eager_op_f32_dot_general_decode_launch-only-floor | #2079 | launch-only-floor | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | — | — | — | — | FAILED |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 18486.8 | 285.7 | 1.7% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 29747.7 | 354.4 | 0.8% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 17974.5 | 538.3 | 2.8% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 29405.9 | 575.9 | 2.0% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 28216.3 | 1206.1 | 2.6% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 29215.3 | 168.1 | 1.6% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 594425.0 | 353.9 | 0.1% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 594927.5 | 2714.9 | 0.5% | ok |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4742509.0 | 15755.5 | 0.4% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4764669.0 | 11245.5 | 0.5% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6735305.0 | 31427.0 | 1.7% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3541938.0 | 6935.0 | 0.2% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41243560.0 | 27796.0 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41326742.0 | 65741.5 | 0.1% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32825498.0 | 165180.0 | 0.4% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9399064.0 | 16040.5 | 0.1% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 389359268.0 | 156079.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 389231459.0 | 166346.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 263115551.0 | 263374.0 | 0.1% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 62092024.0 | 78228.0 | 0.2% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 318393.1 | 6976.5 | 1.9% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 316454.4 | 3627.1 | 2.0% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 199922.9 | 475.3 | 0.2% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 436274.5 | 2750.7 | 0.8% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 440283.4 | 3079.4 | 0.8% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 403758.5 | 4525.8 | 1.1% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 2 | 1968065.5 | 14155.5 | 0.4% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 1 | 1979856.0 | 9035.0 | 0.6% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 20011298.0 | 116484.5 | 0.6% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 324948.4 | 7839.0 | 1.5% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 332447.4 | 1585.1 | 0.3% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 261028.8 | 1701.3 | 0.6% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 481376.0 | 4730.8 | 1.0% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 477660.9 | 1040.0 | 0.2% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 655909.5 | 5837.8 | 2.1% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3338531.0 | 59572.0 | 1.6% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3283159.0 | 11560.5 | 0.3% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49951487.0 | 304840.0 | 0.4% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 54703.9 | 166.4 | 0.3% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 256 | 9426.7 | 1056.0 | 18.4% | NOISY |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 512 | 7810.5 | 23.0 | 0.3% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 2 | 8990.5 | 455.0 | 4.1% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 32616025.0 | 99444.0 | 0.5% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 10939.1 | 304.5 | 2.0% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 29781.3 | 862.0 | 1.9% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 4 | 534270.5 | 9335.4 | 1.3% | ok |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 64279.0 | 2835.0 | 6.7% | ok |
| vendor_handle_cublas_f64_runtime-handle | #2083 | runtime-handle | tenferro-cuda | f64 |  | 1 | 32 | 65353.8 | 2130.4 | 2.7% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so a sample is exactly one call: the median is that call's latency, not an average over a batch, and the precondition sits outside its clock.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 102754.0 | 7944.0 | 5.4% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 40471.0 | 1700.5 | 7.0% | ok |
| contraction_plan_churn_f64_below-capacity-varying-workspace | #1873 #1805 | below-capacity-varying-workspace | tenferro-cuda | f64 | distinct=48, n=64 | 1 | 1 | 40631.0 | 1080.0 | 3.1% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 54402.0 | 939.5 | 5.7% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 51181.0 | 975.0 | 1.1% | ok |
| borrowed_input_gemm_f64_n8_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 39742.0 | 5235.0 | 10.5% | NOISY |
| borrowed_input_gemm_f64_n8_view-input | #1925 | view-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 49691.0 | 2259.5 | 3.6% | ok |
| borrowed_input_gemm_f64_n32_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 40402.0 | 1205.0 | 2.4% | ok |
| borrowed_input_gemm_f64_n32_view-input | #1925 | view-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 38301.0 | 5359.5 | 9.4% | ok |

## State diagnostics (no timing)

Allocation and lifetime state, read before and after each arm: driver free device bytes for the pool, and process RSS where the issue's observable is host memory (#1924). Bytes, not nanoseconds; a case that keeps memory it no longer needs shows up as a non-zero retained or growth value, and a probe that the driver refuses shows up in `probe_status`.

### vendor_churn_f64_cycles8 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 8 | — |
| free_bytes_after | 84472037376 | 80558.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 33554432 | 32.0 |
| rss_bytes_after | 669179904 | 638.2 |
| rss_bytes_before | 654737408 | 624.4 |
| rss_growth_bytes | 14442496 | 13.8 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 84505591808 | 654745600 |
| 1 | 0 | 84505591808 | 669163520 |
| 2 | -33554432 | 84472037376 | 669171712 |
| 3 | 0 | 84472037376 | 669171712 |
| 4 | 0 | 84472037376 | 669171712 |
| 5 | 0 | 84472037376 | 669171712 |
| 6 | 0 | 84472037376 | 669179904 |
| 7 | 0 | 84472037376 | 669179904 |

### vendor_churn_f64_cycles64 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 64 | — |
| free_bytes_after | 84438482944 | 80526.8 |
| free_bytes_before | 84505591808 | 80590.8 |
| retained_bytes_over_cycles | 67108864 | 64.0 |
| rss_bytes_after | 669102080 | 638.1 |
| rss_bytes_before | 654548992 | 624.2 |
| rss_growth_bytes | 14553088 | 13.9 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 84505591808 | 654557184 |
| 1 | 0 | 84505591808 | 668999680 |
| 2 | -33554432 | 84472037376 | 669003776 |
| 3 | 0 | 84472037376 | 669003776 |
| 4 | 0 | 84472037376 | 669003776 |
| 5 | 0 | 84472037376 | 669003776 |
| 6 | 0 | 84472037376 | 669011968 |
| 7 | 0 | 84472037376 | 669011968 |
| 8 | 0 | 84472037376 | 669024256 |
| 9 | 0 | 84472037376 | 669024256 |
| 10 | 0 | 84472037376 | 669024256 |
| 11 | 0 | 84472037376 | 669028352 |
| 12 | 0 | 84472037376 | 669028352 |
| 13 | 0 | 84472037376 | 669028352 |
| 14 | 0 | 84472037376 | 669032448 |
| 15 | 0 | 84472037376 | 669032448 |
| 16 | 0 | 84472037376 | 669032448 |
| 17 | 0 | 84472037376 | 669032448 |
| 18 | 0 | 84472037376 | 669036544 |
| 19 | 0 | 84472037376 | 669036544 |
| 20 | 0 | 84472037376 | 669040640 |
| 21 | 0 | 84472037376 | 669040640 |
| 22 | 0 | 84472037376 | 669040640 |
| 23 | 0 | 84472037376 | 669048832 |
| 24 | 0 | 84472037376 | 669048832 |
| 25 | 0 | 84472037376 | 669048832 |
| 26 | 0 | 84472037376 | 669048832 |
| 27 | 0 | 84472037376 | 669048832 |
| 28 | 0 | 84472037376 | 669048832 |
| 29 | 0 | 84472037376 | 669048832 |
| 30 | 0 | 84472037376 | 669061120 |
| 31 | 0 | 84472037376 | 669065216 |
| 32 | 0 | 84472037376 | 669065216 |
| 33 | 0 | 84472037376 | 669065216 |
| 34 | -33554432 | 84438482944 | 669069312 |
| 35 | 0 | 84438482944 | 669069312 |
| 36 | 0 | 84438482944 | 669069312 |
| 37 | 0 | 84438482944 | 669069312 |
| 38 | 0 | 84438482944 | 669073408 |
| 39 | 0 | 84438482944 | 669081600 |
| 40 | 0 | 84438482944 | 669081600 |
| 41 | 0 | 84438482944 | 669081600 |
| 42 | 0 | 84438482944 | 669081600 |
| 43 | 0 | 84438482944 | 669081600 |
| 44 | 0 | 84438482944 | 669081600 |
| 45 | 0 | 84438482944 | 669081600 |
| 46 | 0 | 84438482944 | 669081600 |
| 47 | 0 | 84438482944 | 669081600 |
| 48 | 0 | 84438482944 | 669081600 |
| 49 | 0 | 84438482944 | 669085696 |
| 50 | 0 | 84438482944 | 669089792 |
| 51 | 0 | 84438482944 | 669089792 |
| 52 | 0 | 84438482944 | 669089792 |
| 53 | 0 | 84438482944 | 669102080 |
| 54 | 0 | 84438482944 | 669102080 |
| 55 | 0 | 84438482944 | 669102080 |
| 56 | 0 | 84438482944 | 669102080 |
| 57 | 0 | 84438482944 | 669102080 |
| 58 | 0 | 84438482944 | 669102080 |
| 59 | 0 | 84438482944 | 669102080 |
| 60 | 0 | 84438482944 | 669102080 |
| 61 | 0 | 84438482944 | 669102080 |
| 62 | 0 | 84438482944 | 669102080 |
| 63 | 0 | 84438482944 | 669102080 |

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

### pool_retention_f64_trim-after-drop (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 76043583488 | 72520.8 |
| free_bytes_after_recreate | 76043583488 | 72520.8 |
| free_bytes_start | 84633518080 | 80712.8 |
| free_bytes_while_alive | 76043583488 | 72520.8 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (trim-after-drop arm)` | — |
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

Median over 22 distinct layouts: first call 0.087 ms, second call 0.042 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 321.267 | 0.089 |
| [2, 5] -> transpose |  | 0.095 | 0.067 |
| [2, 7] -> transpose |  | 0.097 | 0.050 |
| [3, 5] -> transpose |  | 0.088 | 0.046 |
| [3, 7] -> transpose |  | 0.091 | 0.046 |
| [4, 5] -> transpose |  | 0.088 | 0.041 |
| [4, 7] -> transpose |  | 0.091 | 0.043 |
| [5, 5] -> transpose |  | 0.091 | 0.048 |
| [5, 7] -> transpose |  | 0.086 | 0.041 |
| [6, 5] -> transpose |  | 0.085 | 0.040 |
| [6, 7] -> transpose |  | 0.082 | 0.041 |
| [7, 5] -> transpose |  | 0.088 | 0.042 |
| [7, 7] -> transpose |  | 0.083 | 0.040 |
| [8, 5] -> transpose |  | 0.080 | 0.042 |
| [8, 7] -> transpose |  | 0.091 | 0.042 |
| [9, 5] -> transpose |  | 0.090 | 0.052 |
| [9, 7] -> transpose |  | 0.087 | 0.042 |
| [10, 5] -> transpose |  | 0.085 | 0.043 |
| [10, 7] -> transpose |  | 0.079 | 0.041 |
| [11, 5] -> transpose |  | 0.081 | 0.041 |
| [11, 7] -> transpose |  | 0.087 | 0.041 |
| [12, 5] -> transpose |  | 0.084 | 0.041 |
| [12, 7] -> transpose |  | 0.082 | 0.041 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.031 ms, second call 0.027 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 53.605 | 0.052 |
| [2, 5] -> transpose |  | 0.046 | 0.040 |
| [2, 7] -> transpose |  | 0.044 | 0.034 |
| [3, 5] -> transpose |  | 0.039 | 0.036 |
| [3, 7] -> transpose |  | 0.039 | 0.035 |
| [4, 5] -> transpose |  | 0.036 | 0.026 |
| [4, 7] -> transpose |  | 0.030 | 0.027 |
| [5, 5] -> transpose |  | 0.031 | 0.029 |
| [5, 7] -> transpose |  | 0.036 | 0.027 |
| [6, 5] -> transpose |  | 0.031 | 0.026 |
| [6, 7] -> transpose |  | 0.031 | 0.027 |
| [7, 5] -> transpose |  | 0.030 | 0.027 |
| [7, 7] -> transpose |  | 0.031 | 0.027 |
| [8, 5] -> transpose |  | 0.029 | 0.027 |
| [8, 7] -> transpose |  | 0.031 | 0.026 |
| [9, 5] -> transpose |  | 0.031 | 0.026 |
| [9, 7] -> transpose |  | 0.030 | 0.026 |
| [10, 5] -> transpose |  | 0.029 | 0.025 |
| [10, 7] -> transpose |  | 0.030 | 0.027 |
| [11, 5] -> transpose |  | 0.029 | 0.026 |
| [11, 7] -> transpose |  | 0.030 | 0.032 |
| [12, 5] -> transpose |  | 0.030 | 0.026 |
| [12, 7] -> transpose |  | 0.030 | 0.025 |


## Timing boundaries

- **alloc_zero / alloc-fill-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_output + fill_zero_write (device fill), device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_zero_output, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: one with_cubecl entry/exit per batch, alloc_zero_output x iterations, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **borrowed_input_gemm / owned-input** — inside: one small dot_general_read with owned inputs, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **borrowed_input_gemm / view-input** — inside: one small dot_general_read with a borrowed transposed input view, one isolated call per sample; device synchronize after the call; outside: backend construction, operand upload, correctness check (download).
- **contraction_plan_churn / below-capacity-varying-workspace** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / distinct-plans** — inside: one re-run of the first contraction after the plan cache was filled by distinct shapes, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **contraction_plan_churn / repeated-plan** — inside: one re-run of the same contraction, one isolated call per sample; device synchronize after the call; outside: backend construction, operand construction and upload, the churn that fills (or repeats) the plan cache, and the synchronize after it, correctness check (download).
- **eager_op / eager-per-call** — inside: one eager op with a with_eager_session entry/exit per operation, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session** — inside: one eager op in a reused with_eager_session, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / eager-session-streaming** — inside: one eager op in a reused with_eager_session, result dropped per op, one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
- **eager_op / launch-only-floor** — inside: one add_read_into on a reused output (no per-op allocation), one device synchronize at the batch edge; outside: backend, runtime and input construction (constant_from_host), correctness check (download).
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
- **pool_retention / trim-after-drop** — inside: —; outside: everything (a state diagnostic, no timing).
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

### FAILED: eager_op_f32_dot_general_decode_launch-only-floor (1 threads)
```
Error: "unknown #2079 floor op dot_general"
```

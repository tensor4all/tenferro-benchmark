# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261012_002939`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261012_002939/run_t1.yaml`

```yaml
timestamp: '2026-10-12T00:29:41Z'
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
    BENCHMARK_COMMIT: 61a96ca14cd311cce50de34d70e874e3c8a9a48c
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.97.1 (8bab26f4f 2026-07-14)
    BENCH_INSTANCE: null
    BENCHMARK_ALLOW_BUSY_GPU: null
  gpu_compute_apps_at_start: []
  gpu_compute_apps_at_end: []
```

## Case status

- Manifest: `gpu/perf_issues` version 2, coverage `full`, effort `standard`
- Expected 139, selected 139, executed 139, unsupported 0, failed 0, missing 0, noisy 4.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `alloc_zero_f64_host-zero-upload_32768el@t1`, `reshape_constant_f32_64M_op-output@t1`, `borrowed_input_gemm_f64_n8_owned-input@t1`, `borrowed_input_gemm_f64_n8_view-input@t1`

Numerical checks: **139/139 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 22414.2 | 2049.2 | 5.1% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6426.8 | 169.0 | 2.3% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9946.0 | 82.9 | 0.6% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 23460.9 | 244.7 | 1.6% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 6982.6 | 60.0 | 0.7% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 58211.7 | 1115.6 | 1.4% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 34290.1 | 225.7 | 0.7% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 13705.5 | 18.3 | 0.3% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12720.2 | 247.0 | 1.4% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 258724.0 | 1175.0 | 0.5% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 89556.1 | 668.1 | 1.0% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50018.8 | 73.0 | 0.2% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 2 | 1252377.5 | 2947.5 | 0.2% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 756640.0 | 1727.6 | 1.2% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636076.2 | 288.8 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 21891632.0 | 34605.0 | 0.1% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 13389609.0 | 190641.0 | 1.5% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10016019.0 | 6175.5 | 0.8% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 21154.0 | 37.9 | 0.2% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7588.3 | 57.8 | 0.7% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 8895.5 | 141.3 | 1.3% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 29072.6 | 30.3 | 0.1% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8096.8 | 47.0 | 0.5% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9676.3 | 138.7 | 0.9% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 34963.5 | 1411.7 | 3.4% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 12980.2 | 67.7 | 0.6% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11833.5 | 232.4 | 2.0% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 616329.8 | 83630.1 | 6.9% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 95784.8 | 350.7 | 0.3% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49584.5 | 118.1 | 0.3% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 8998114.0 | 491092.2 | 3.2% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 987527.2 | 14308.0 | 1.9% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 656297.2 | 308.5 | 0.0% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 146149711.0 | 2312889.0 | 1.7% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 15587886.0 | 173574.5 | 2.0% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10336359.0 | 2985.0 | 0.0% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 2151.7 | 38.1 | 2.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 25046.0 | 768.6 | 4.1% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 4116.4 | 125.3 | 2.2% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 24763.6 | 1470.9 | 3.8% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 19050.3 | 134.7 | 0.5% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 40397.9 | 163.3 | 0.3% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 512 | 6830.8 | 112.0 | 3.6% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 42200.0 | 4256.6 | 5.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 19900.8 | 292.1 | 0.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 41349.8 | 3201.0 | 4.7% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 512 | 7337.7 | 477.0 | 3.7% | ok |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 40984.0 | 2385.9 | 5.3% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 20682.6 | 93.3 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 41976.6 | 143.4 | 0.3% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 58498.6 | 13707.0 | 29.7% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 42284.2 | 966.2 | 2.2% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 19427.5 | 939.1 | 3.9% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 43098.8 | 285.8 | 0.5% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 2 | 1095208.0 | 2850.0 | 0.3% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 44428.2 | 179.1 | 0.5% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 6834516.0 | 540391.5 | 5.5% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 17735.6 | 54.7 | 0.3% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 6396411.0 | 122488.0 | 2.0% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 18365.0 | 758.9 | 3.7% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 8176275.0 | 354281.0 | 2.5% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 22571.4 | 512.8 | 2.1% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 6962319.0 | 138964.0 | 2.1% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 128 | 20904.9 | 41.1 | 0.1% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5707302.0 | 59142.0 | 0.8% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2606799.0 | 9070.0 | 0.3% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 110385867.0 | 1229622.0 | 1.1% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 36256189.0 | 48611.5 | 0.1% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17684982.0 | 114138.5 | 1.1% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14350382.0 | 76443.0 | 0.5% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 300131159.0 | 1731855.5 | 1.2% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 229266622.0 | 1844165.5 | 0.5% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 385359.1 | 1045.0 | 0.3% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 369466.1 | 11736.6 | 3.3% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 4 | 539916.2 | 9024.0 | 3.2% | ok |
| eager_op_f32_add_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 10464.3 | 89.0 | 0.9% | ok |
| eager_op_f32_dot_general_decode_eager-session | #2079 | eager-session | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 64 | 32484.7 | 278.5 | 0.8% | ok |
| gather_embedding_f32_vocab8192_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 32 | 95856.3 | 453.8 | 0.5% | ok |
| gather_embedding_f32_vocab8192_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=8192, hidden=1024, tokens=8 | 1 | 128 | 24558.5 | 169.1 | 0.4% | ok |
| gather_embedding_f32_vocab65536_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 4 | 607041.0 | 3356.4 | 0.4% | ok |
| gather_embedding_f32_vocab65536_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=65536, hidden=1024, tokens=8 | 1 | 128 | 29183.0 | 1203.1 | 5.6% | ok |
| gather_embedding_f32_vocab248320_eager-gather | #2080 | eager-gather | tenferro-cuda | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 1 | 2247198.0 | 10321.0 | 0.4% | ok |
| gather_embedding_f32_vocab248320_cudarc-row-copies | #2080 | cudarc-row-copies | cudarc | f32 | vocab=248320, hidden=1024, tokens=8 | 1 | 128 | 29437.3 | 88.2 | 0.3% | ok |
| reshape_constant_f32_64M_constant | #2082 | constant | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 647264.8 | 56609.0 | 4.9% | ok |
| reshape_constant_f32_64M_op-output | #2082 | op-output | tenferro-cuda | f32 | elements=67108864, rows=65536, cols=1024 | 1 | 4 | 593485.2 | 120611.0 | 19.1% | NOISY |
| svd_driver_c64_n64_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=64, driver=auto | 1 | 1 | 4785085.0 | 17770.0 | 0.3% | ok |
| svd_driver_c64_n64_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=64, driver=gesvdj | 1 | 1 | 4777475.0 | 51421.5 | 1.1% | ok |
| svd_driver_c64_n64_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=64, driver=gesvd | 1 | 1 | 6773016.0 | 13155.5 | 0.2% | ok |
| svd_driver_c64_n64_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=64, driver=xgesvdp | 1 | 1 | 3582469.0 | 10730.5 | 0.4% | ok |
| svd_driver_c64_n256_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=256, driver=auto | 1 | 1 | 41357806.0 | 23515.5 | 0.0% | ok |
| svd_driver_c64_n256_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=256, driver=gesvdj | 1 | 1 | 41430658.0 | 41957.0 | 0.1% | ok |
| svd_driver_c64_n256_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=256, driver=gesvd | 1 | 1 | 32416425.0 | 97838.5 | 0.2% | ok |
| svd_driver_c64_n256_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=256, driver=xgesvdp | 1 | 1 | 9346264.0 | 12955.0 | 0.1% | ok |
| svd_driver_c64_n1024_auto | #1839 #1849 | auto | tenferro-cuda | c64 | n=1024, driver=auto | 1 | 1 | 388827958.0 | 41704.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvdj | #1839 #1849 | gesvdj | tenferro-cuda | c64 | n=1024, driver=gesvdj | 1 | 1 | 388851994.0 | 43163.5 | 0.0% | ok |
| svd_driver_c64_n1024_gesvd | #1839 #1849 | gesvd | tenferro-cuda | c64 | n=1024, driver=gesvd | 1 | 1 | 263066552.0 | 42052.5 | 0.0% | ok |
| svd_driver_c64_n1024_xgesvdp | #1839 #1849 | xgesvdp | tenferro-cuda | c64 | n=1024, driver=xgesvdp | 1 | 1 | 62329319.0 | 73133.0 | 0.1% | ok |
| eigh_batched_f64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=16, batch=64, driver=auto | 1 | 8 | 306324.4 | 11902.2 | 5.7% | ok |
| eigh_batched_f64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=16, batch=64, driver=syevd | 1 | 8 | 300569.2 | 3840.1 | 1.2% | ok |
| eigh_batched_f64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=16, batch=64, driver=syevj | 1 | 16 | 209958.3 | 5883.3 | 2.2% | ok |
| eigh_batched_f64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=32, batch=64, driver=auto | 1 | 8 | 453053.8 | 3115.2 | 0.6% | ok |
| eigh_batched_f64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=32, batch=64, driver=syevd | 1 | 8 | 432590.6 | 9412.8 | 1.2% | ok |
| eigh_batched_f64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=32, batch=64, driver=syevj | 1 | 8 | 423061.8 | 5370.1 | 1.2% | ok |
| eigh_batched_f64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | f64 | n=128, batch=64, driver=auto | 1 | 1 | 2011981.0 | 19026.0 | 0.8% | ok |
| eigh_batched_f64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | f64 | n=128, batch=64, driver=syevd | 1 | 1 | 2078974.0 | 18295.5 | 1.2% | ok |
| eigh_batched_f64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | f64 | n=128, batch=64, driver=syevj | 1 | 1 | 19910998.0 | 19280.5 | 0.1% | ok |
| eigh_batched_c64_n16_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=16, batch=64, driver=auto | 1 | 8 | 330777.6 | 1125.1 | 0.5% | ok |
| eigh_batched_c64_n16_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=16, batch=64, driver=syevd | 1 | 8 | 324867.5 | 1346.3 | 0.4% | ok |
| eigh_batched_c64_n16_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=16, batch=64, driver=syevj | 1 | 8 | 269375.6 | 700.1 | 0.3% | ok |
| eigh_batched_c64_n32_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=32, batch=64, driver=auto | 1 | 8 | 484318.5 | 1650.1 | 0.3% | ok |
| eigh_batched_c64_n32_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=32, batch=64, driver=syevd | 1 | 8 | 483359.8 | 3835.1 | 0.7% | ok |
| eigh_batched_c64_n32_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=32, batch=64, driver=syevj | 1 | 4 | 669812.8 | 2164.0 | 0.5% | ok |
| eigh_batched_c64_n128_auto | #1846 #1852 #1923 | auto | tenferro-cuda | c64 | n=128, batch=64, driver=auto | 1 | 1 | 3334102.0 | 33067.0 | 0.9% | ok |
| eigh_batched_c64_n128_syevd | #1846 #1852 #1923 | syevd | tenferro-cuda | c64 | n=128, batch=64, driver=syevd | 1 | 1 | 3316521.0 | 13445.5 | 0.6% | ok |
| eigh_batched_c64_n128_syevj | #1846 #1852 #1923 | syevj | tenferro-cuda | c64 | n=128, batch=64, driver=syevj | 1 | 1 | 49782140.0 | 39911.5 | 0.1% | ok |
| hadamard_f32_1024x1024_dot-general | #1898 | dot-general | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 64 | 54532.9 | 92.6 | 0.1% | ok |
| hadamard_f32_1024x1024_multiply | #1898 | multiply | tenferro-cuda | f32 | rows=1024, cols=1024 | 1 | 256 | 7668.1 | 602.5 | 6.2% | ok |

## Per-call session entry diagnostics (not operation comparisons)

The measured call includes session/scope entry or exit: the eager-per-call arms enter and leave an eager session per operation, the raw-scope arms enter and leave a raw CUDA scope (or time the explicit barrier they replace), and the vendor arms enter a raw scope per operation. Each case's timing note lists its own boundary.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| raw_scope_entry_f64_empty-idle | #2078 | empty-idle | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 256 | 13131.3 | 67.0 | 1.6% | ok |
| raw_scope_entry_f64_empty-behind-queued-gemm | #2078 | empty-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 32993644.0 | 17225.5 | 0.3% | ok |
| raw_scope_entry_f64_sync-behind-queued-gemm | #2078 | sync-behind-queued-gemm | tenferro-cuda | f64 | gemm_n=4096, queued_gemms=4 | 1 | 1 | 33020296.0 | 623208.5 | 7.6% | ok |
| eager_op_f32_add_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | hidden=1024, tokens=8, op=add | 1 | 256 | 11300.8 | 88.5 | 0.8% | ok |
| eager_op_f32_dot_general_decode_eager-per-call | #2079 | eager-per-call | tenferro-cuda | f32 | in=1024, out=1024, len=8, op=dot_general | 1 | 128 | 28633.1 | 1055.4 | 3.9% | ok |
| vendor_handle_cublas_f64_create-destroy | #2083 | create-destroy | tenferro-cuda | f64 |  | 1 | 4 | 563649.5 | 8156.4 | 1.4% | ok |
| vendor_handle_cublas_f64_reused | #2083 | reused | tenferro-cuda | f64 |  | 1 | 32 | 72381.6 | 1590.7 | 2.6% | ok |

## Issue-workload diagnostics (not steady-state operation rows)

The workload of the issue needs a per-call precondition that cannot be batched (a cache miss that reconstructs the contraction plan and its workspace, or a caller-provided destination handle built per call), so a sample is exactly one call: the median is that call's latency, not an average over a batch, and the precondition sits outside its clock.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| contraction_plan_churn_f64_distinct-plans | #1873 #1805 | distinct-plans | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 85723.0 | 1164.5 | 5.3% | ok |
| contraction_plan_churn_f64_repeated-plan | #1873 #1805 | repeated-plan | tenferro-cuda | f64 | distinct=72, n=64 | 1 | 1 | 39961.0 | 315.5 | 1.2% | ok |
| dot_general_destination_f64_owned-dst | #1949 | owned-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 50771.0 | 1799.5 | 2.8% | ok |
| dot_general_destination_f64_borrowed-dst | #1949 | borrowed-dst | tenferro-cuda | f64 | n=512 | 1 | 1 | 48951.0 | 1870.5 | 3.2% | ok |
| borrowed_input_gemm_f64_n8_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 41731.0 | 2259.5 | 17.7% | NOISY |
| borrowed_input_gemm_f64_n8_view-input | #1925 | view-input | tenferro-cuda | f64 | n=8 | 1 | 1 | 41732.0 | 1700.5 | 17.2% | NOISY |
| borrowed_input_gemm_f64_n32_owned-input | #1925 | owned-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 43831.0 | 1271.0 | 1.8% | ok |
| borrowed_input_gemm_f64_n32_view-input | #1925 | view-input | tenferro-cuda | f64 | n=32 | 1 | 1 | 38961.0 | 1035.0 | 2.1% | ok |

## State diagnostics (no timing)

Allocation and lifetime state, read before and after each arm: driver free device bytes for the pool, and process RSS where the issue's observable is host memory (#1924). Bytes, not nanoseconds; a case that keeps memory it no longer needs shows up as a non-zero retained or growth value, and a probe that the driver refuses shows up in `probe_status`.

### vendor_churn_f64_cycles8 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 8 | — |
| free_bytes_after | 83929333760 | 80041.2 |
| free_bytes_before | 83929333760 | 80041.2 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 666464256 | 635.6 |
| rss_bytes_before | 652410880 | 622.2 |
| rss_growth_bytes | 14053376 | 13.4 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 83929333760 | 652410880 |
| 1 | 0 | 83929333760 | 666456064 |
| 2 | 0 | 83929333760 | 666464256 |
| 3 | 0 | 83929333760 | 666464256 |
| 4 | 0 | 83929333760 | 666464256 |
| 5 | 0 | 83929333760 | 666464256 |
| 6 | 0 | 83929333760 | 666464256 |
| 7 | 0 | 83929333760 | 666464256 |

### vendor_churn_f64_cycles64 (#1924)

| Field | Value | MiB |
|---|---:|---:|
| cycles | 64 | — |
| free_bytes_after | 83929333760 | 80041.2 |
| free_bytes_before | 83929333760 | 80041.2 |
| retained_bytes_over_cycles | 0 | 0.0 |
| rss_bytes_after | 666677248 | 635.8 |
| rss_bytes_before | 652574720 | 622.3 |
| rss_growth_bytes | 14102528 | 13.4 |

| Cycle | delta_bytes | free_bytes | rss_bytes |
|---:|---:|---:|---:|
| 0 | 0 | 83929333760 | 652574720 |
| 1 | 0 | 83929333760 | 666619904 |
| 2 | 0 | 83929333760 | 666628096 |
| 3 | 0 | 83929333760 | 666628096 |
| 4 | 0 | 83929333760 | 666628096 |
| 5 | 0 | 83929333760 | 666628096 |
| 6 | 0 | 83929333760 | 666628096 |
| 7 | 0 | 83929333760 | 666628096 |
| 8 | 0 | 83929333760 | 666628096 |
| 9 | 0 | 83929333760 | 666628096 |
| 10 | 0 | 83929333760 | 666628096 |
| 11 | 0 | 83929333760 | 666632192 |
| 12 | 0 | 83929333760 | 666632192 |
| 13 | 0 | 83929333760 | 666636288 |
| 14 | 0 | 83929333760 | 666636288 |
| 15 | 0 | 83929333760 | 666636288 |
| 16 | 0 | 83929333760 | 666636288 |
| 17 | 0 | 83929333760 | 666636288 |
| 18 | 0 | 83929333760 | 666640384 |
| 19 | 0 | 83929333760 | 666648576 |
| 20 | 0 | 83929333760 | 666648576 |
| 21 | 0 | 83929333760 | 666648576 |
| 22 | 0 | 83929333760 | 666648576 |
| 23 | 0 | 83929333760 | 666652672 |
| 24 | 0 | 83929333760 | 666652672 |
| 25 | 0 | 83929333760 | 666652672 |
| 26 | 0 | 83929333760 | 666652672 |
| 27 | 0 | 83929333760 | 666652672 |
| 28 | 0 | 83929333760 | 666652672 |
| 29 | 0 | 83929333760 | 666652672 |
| 30 | 0 | 83929333760 | 666652672 |
| 31 | 0 | 83929333760 | 666656768 |
| 32 | 0 | 83929333760 | 666656768 |
| 33 | 0 | 83929333760 | 666656768 |
| 34 | 0 | 83929333760 | 666656768 |
| 35 | 0 | 83929333760 | 666656768 |
| 36 | 0 | 83929333760 | 666656768 |
| 37 | 0 | 83929333760 | 666669056 |
| 38 | 0 | 83929333760 | 666669056 |
| 39 | 0 | 83929333760 | 666669056 |
| 40 | 0 | 83929333760 | 666669056 |
| 41 | 0 | 83929333760 | 666669056 |
| 42 | 0 | 83929333760 | 666673152 |
| 43 | 0 | 83929333760 | 666673152 |
| 44 | 0 | 83929333760 | 666673152 |
| 45 | 0 | 83929333760 | 666673152 |
| 46 | 0 | 83929333760 | 666673152 |
| 47 | 0 | 83929333760 | 666673152 |
| 48 | 0 | 83929333760 | 666673152 |
| 49 | 0 | 83929333760 | 666673152 |
| 50 | 0 | 83929333760 | 666673152 |
| 51 | 0 | 83929333760 | 666673152 |
| 52 | 0 | 83929333760 | 666673152 |
| 53 | 0 | 83929333760 | 666673152 |
| 54 | 0 | 83929333760 | 666673152 |
| 55 | 0 | 83929333760 | 666673152 |
| 56 | 0 | 83929333760 | 666673152 |
| 57 | 0 | 83929333760 | 666673152 |
| 58 | 0 | 83929333760 | 666673152 |
| 59 | 0 | 83929333760 | 666673152 |
| 60 | 0 | 83929333760 | 666673152 |
| 61 | 0 | 83929333760 | 666677248 |
| 62 | 0 | 83929333760 | 666677248 |
| 63 | 0 | 83929333760 | 666677248 |

### pool_retention_f64_alloc-free (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 75467325440 | 71971.2 |
| free_bytes_after_recreate | 75467325440 | 71971.2 |
| free_bytes_start | 84057260032 | 80163.2 |
| free_bytes_while_alive | 75467325440 | 71971.2 |
| peak_used_bytes | 8589934592 | 8192.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (alloc-free arm)` | — |
| retained_bytes_after_drop | 8589934592 | 8192.0 |
| tensor_gib | 1 | — |
| tensors | 8 | — |

### pool_retention_f64_runtime-recreate (#2081)

| Field | Value | MiB |
|---|---:|---:|
| free_bytes_after_drop | 74997563392 | 71523.2 |
| free_bytes_after_recreate | 74997563392 | 71523.2 |
| free_bytes_start | 83721715712 | 79843.2 |
| free_bytes_while_alive | 74997563392 | 71523.2 |
| peak_used_bytes | 8724152320 | 8320.0 |
| probe_gib | 6 | — |
| probe_status | `not requested (runtime-recreate arm)` | — |
| retained_bytes_after_drop | 8724152320 | 8320.0 |
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

Median over 22 distinct layouts: first call 0.073 ms, second call 0.037 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 294.556 | 0.087 |
| [2, 5] -> transpose |  | 0.089 | 0.044 |
| [2, 7] -> transpose |  | 0.074 | 0.051 |
| [3, 5] -> transpose |  | 0.073 | 0.035 |
| [3, 7] -> transpose |  | 0.072 | 0.036 |
| [4, 5] -> transpose |  | 0.079 | 0.038 |
| [4, 7] -> transpose |  | 0.070 | 0.036 |
| [5, 5] -> transpose |  | 0.076 | 0.036 |
| [5, 7] -> transpose |  | 0.074 | 0.037 |
| [6, 5] -> transpose |  | 0.072 | 0.037 |
| [6, 7] -> transpose |  | 0.072 | 0.044 |
| [7, 5] -> transpose |  | 0.075 | 0.036 |
| [7, 7] -> transpose |  | 0.071 | 0.036 |
| [8, 5] -> transpose |  | 0.071 | 0.037 |
| [8, 7] -> transpose |  | 0.077 | 0.037 |
| [9, 5] -> transpose |  | 0.068 | 0.043 |
| [9, 7] -> transpose |  | 0.076 | 0.036 |
| [10, 5] -> transpose |  | 0.073 | 0.036 |
| [10, 7] -> transpose |  | 0.070 | 0.036 |
| [11, 5] -> transpose |  | 0.069 | 0.037 |
| [11, 7] -> transpose |  | 0.068 | 0.037 |
| [12, 5] -> transpose |  | 0.074 | 0.037 |
| [12, 7] -> transpose |  | 0.073 | 0.036 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.040 ms, second call 0.035 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 47.063 | 0.052 |
| [2, 5] -> transpose |  | 0.045 | 0.038 |
| [2, 7] -> transpose |  | 0.039 | 0.035 |
| [3, 5] -> transpose |  | 0.038 | 0.033 |
| [3, 7] -> transpose |  | 0.050 | 0.037 |
| [4, 5] -> transpose |  | 0.041 | 0.036 |
| [4, 7] -> transpose |  | 0.040 | 0.036 |
| [5, 5] -> transpose |  | 0.041 | 0.036 |
| [5, 7] -> transpose |  | 0.044 | 0.036 |
| [6, 5] -> transpose |  | 0.041 | 0.036 |
| [6, 7] -> transpose |  | 0.041 | 0.035 |
| [7, 5] -> transpose |  | 0.042 | 0.037 |
| [7, 7] -> transpose |  | 0.042 | 0.037 |
| [8, 5] -> transpose |  | 0.041 | 0.037 |
| [8, 7] -> transpose |  | 0.032 | 0.026 |
| [9, 5] -> transpose |  | 0.029 | 0.030 |
| [9, 7] -> transpose |  | 0.037 | 0.032 |
| [10, 5] -> transpose |  | 0.030 | 0.027 |
| [10, 7] -> transpose |  | 0.031 | 0.028 |
| [11, 5] -> transpose |  | 0.032 | 0.029 |
| [11, 7] -> transpose |  | 0.032 | 0.028 |
| [12, 5] -> transpose |  | 0.032 | 0.028 |
| [12, 7] -> transpose |  | 0.030 | 0.028 |


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

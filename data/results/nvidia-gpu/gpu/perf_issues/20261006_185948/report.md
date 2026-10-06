# GPU Performance-Issue Workloads

- Suite: `gpu/perf_issues`
- Raw run: `data/results/nvidia-gpu/gpu/perf_issues/20261006_185948`

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

Full metadata: `data/results/nvidia-gpu/gpu/perf_issues/20261006_185948/run_t1.yaml`

```yaml
timestamp: '2026-10-06T18:59:50Z'
tenferro_rs:
  path: /workspaces/t4a-2010-bench-gpu/extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/t4a-2010-bench-gpu/extern/tenferro-rs
blas:
  implementation: none
environment:
  hostname: bd10f5edf7ea
  os: Linux-6.8.0-101-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    BENCHMARK_COMMIT: 320964928fdd481aaaf81ceae9606c0ea8df559f
    OPENBLAS_ROOT: /opt/openblas
    CUDA_HOME: /usr/local/cuda
    USE_CUDA: '1'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `gpu/perf_issues` version 1, coverage `full`, effort `standard`
- Expected 77, selected 77, executed 77, unsupported 0, failed 0, missing 0, noisy 2.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `alloc_zero_f64_host-zero-upload_512el@t1`, `alloc_zero_f64_host-zero-upload_32768el@t1`

Numerical checks: **77/77 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| transfer_up_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=up | 1 | 128 | 19688.9 | 132.4 | 0.6% | ok |
| transfer_up_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=up | 1 | 512 | 6400.5 | 16.0 | 1.0% | ok |
| transfer_up_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=up | 1 | 256 | 9922.4 | 71.8 | 1.0% | ok |
| transfer_up_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=up | 1 | 128 | 24496.7 | 130.7 | 0.7% | ok |
| transfer_up_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=up | 1 | 512 | 7600.5 | 222.6 | 2.2% | ok |
| transfer_up_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=up | 1 | 64 | 61257.9 | 769.2 | 1.0% | ok |
| transfer_up_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=up | 1 | 64 | 33773.3 | 353.8 | 0.7% | ok |
| transfer_up_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12769.4 | 99.8 | 1.0% | ok |
| transfer_up_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=up | 1 | 256 | 12716.5 | 134.3 | 0.9% | ok |
| transfer_up_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=up | 1 | 8 | 244219.1 | 2391.3 | 0.8% | ok |
| transfer_up_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=up | 1 | 32 | 89147.6 | 176.5 | 0.2% | ok |
| transfer_up_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=up | 1 | 64 | 50052.1 | 188.0 | 0.3% | ok |
| transfer_up_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=up | 1 | 1 | 3713930.0 | 256973.0 | 8.4% | ok |
| transfer_up_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 753139.5 | 1512.8 | 0.2% | ok |
| transfer_up_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=up | 1 | 4 | 636375.5 | 105.2 | 0.0% | ok |
| transfer_up_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=up | 1 | 1 | 182303567.0 | 4162814.5 | 1.7% | ok |
| transfer_up_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 13321361.0 | 19726.0 | 0.2% | ok |
| transfer_up_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=up | 1 | 1 | 10016514.0 | 5155.0 | 0.8% | ok |
| transfer_down_f64_tenferro_8B | #2009 | tenferro | tenferro-cuda | f64 | bytes=8, direction=down | 1 | 128 | 19339.5 | 87.9 | 3.2% | ok |
| transfer_down_f64_cudarc-pageable_8B | #2009 | cudarc-pageable | cudarc | f64 | bytes=8, direction=down | 1 | 512 | 7448.5 | 61.1 | 1.2% | ok |
| transfer_down_f64_cudarc-pinned_8B | #2009 | cudarc-pinned | cudarc | f64 | bytes=8, direction=down | 1 | 256 | 9610.1 | 44.2 | 1.3% | ok |
| transfer_down_f64_tenferro_4096B | #2009 | tenferro | tenferro-cuda | f64 | bytes=4096, direction=down | 1 | 128 | 30347.6 | 2433.0 | 4.1% | ok |
| transfer_down_f64_cudarc-pageable_4096B | #2009 | cudarc-pageable | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 8156.4 | 313.0 | 2.7% | ok |
| transfer_down_f64_cudarc-pinned_4096B | #2009 | cudarc-pinned | cudarc | f64 | bytes=4096, direction=down | 1 | 256 | 9516.3 | 388.9 | 2.9% | ok |
| transfer_down_f64_tenferro_65536B | #2009 | tenferro | tenferro-cuda | f64 | bytes=65536, direction=down | 1 | 64 | 34006.3 | 4610.6 | 7.4% | ok |
| transfer_down_f64_cudarc-pageable_65536B | #2009 | cudarc-pageable | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 12983.6 | 35.1 | 0.2% | ok |
| transfer_down_f64_cudarc-pinned_65536B | #2009 | cudarc-pinned | cudarc | f64 | bytes=65536, direction=down | 1 | 256 | 11700.3 | 81.0 | 0.8% | ok |
| transfer_down_f64_tenferro_1048576B | #2009 | tenferro | tenferro-cuda | f64 | bytes=1048576, direction=down | 1 | 8 | 612597.4 | 1911.9 | 2.3% | ok |
| transfer_down_f64_cudarc-pageable_1048576B | #2009 | cudarc-pageable | cudarc | f64 | bytes=1048576, direction=down | 1 | 32 | 95071.8 | 81.6 | 0.1% | ok |
| transfer_down_f64_cudarc-pinned_1048576B | #2009 | cudarc-pinned | cudarc | f64 | bytes=1048576, direction=down | 1 | 64 | 49768.8 | 214.5 | 0.4% | ok |
| transfer_down_f64_tenferro_16777216B | #2009 | tenferro | tenferro-cuda | f64 | bytes=16777216, direction=down | 1 | 2 | 8770798.5 | 474422.2 | 3.8% | ok |
| transfer_down_f64_cudarc-pageable_16777216B | #2009 | cudarc-pageable | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 993469.5 | 11774.4 | 1.9% | ok |
| transfer_down_f64_cudarc-pinned_16777216B | #2009 | cudarc-pinned | cudarc | f64 | bytes=16777216, direction=down | 1 | 4 | 655761.2 | 322.6 | 2.5% | ok |
| transfer_down_f64_tenferro_268435456B | #2009 | tenferro | tenferro-cuda | f64 | bytes=268435456, direction=down | 1 | 1 | 144875676.0 | 2586233.0 | 2.3% | ok |
| transfer_down_f64_cudarc-pageable_268435456B | #2009 | cudarc-pageable | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 15099559.0 | 378162.5 | 2.2% | ok |
| transfer_down_f64_cudarc-pinned_268435456B | #2009 | cudarc-pinned | cudarc | f64 | bytes=268435456, direction=down | 1 | 1 | 10325184.0 | 3310.5 | 0.0% | ok |
| alloc_zero_f64_alloc-zero-within-callback_0el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=0 | 1 | 2048 | 3398.5 | 201.6 | 3.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_0el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 22769.4 | 784.1 | 2.7% | ok |
| alloc_zero_f64_host-zero-upload_0el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=0 | 1 | 1024 | 3720.7 | 33.1 | 1.6% | ok |
| alloc_zero_f64_alloc-fill-per-callback_0el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=0 | 1 | 128 | 23129.5 | 1209.4 | 3.7% | ok |
| alloc_zero_f64_alloc-zero-within-callback_16el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=16 | 1 | 128 | 20185.4 | 830.2 | 3.0% | ok |
| alloc_zero_f64_alloc-zero-per-callback_16el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 36928.1 | 195.8 | 0.4% | ok |
| alloc_zero_f64_host-zero-upload_16el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=16 | 1 | 512 | 6595.9 | 100.6 | 3.1% | ok |
| alloc_zero_f64_alloc-fill-per-callback_16el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=16 | 1 | 64 | 35761.9 | 1832.2 | 4.8% | ok |
| alloc_zero_f64_alloc-zero-within-callback_512el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=512 | 1 | 128 | 17650.8 | 141.5 | 0.6% | ok |
| alloc_zero_f64_alloc-zero-per-callback_512el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 41021.8 | 4877.4 | 7.6% | ok |
| alloc_zero_f64_host-zero-upload_512el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=512 | 1 | 256 | 5816.2 | 155.0 | 12.4% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_512el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=512 | 1 | 64 | 38977.2 | 147.1 | 1.3% | ok |
| alloc_zero_f64_alloc-zero-within-callback_32768el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=32768 | 1 | 128 | 19231.6 | 641.3 | 4.4% | ok |
| alloc_zero_f64_alloc-zero-per-callback_32768el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 38951.7 | 2345.2 | 4.1% | ok |
| alloc_zero_f64_host-zero-upload_32768el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=32768 | 1 | 16 | 64812.8 | 1261.3 | 30.6% | NOISY |
| alloc_zero_f64_alloc-fill-per-callback_32768el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=32768 | 1 | 64 | 41828.2 | 203.3 | 0.4% | ok |
| alloc_zero_f64_alloc-zero-within-callback_2097152el | #1887 | alloc-zero-within-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 64 | 19410.3 | 165.3 | 0.7% | ok |
| alloc_zero_f64_alloc-zero-per-callback_2097152el | #1887 | alloc-zero-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 41662.3 | 158.7 | 0.4% | ok |
| alloc_zero_f64_host-zero-upload_2097152el | #1887 | host-zero-upload | tenferro-cuda | f64 | elements=2097152 | 1 | 1 | 3321347.0 | 217437.5 | 5.3% | ok |
| alloc_zero_f64_alloc-fill-per-callback_2097152el | #1887 | alloc-fill-per-callback | tenferro-cuda | f64 | elements=2097152 | 1 | 32 | 42040.4 | 196.2 | 0.5% | ok |
| small_blocks_gemm_f64_loop_n8_count256 | #1885 | loop | tenferro-cuda | f64 | n=8, count=256 | 1 | 1 | 7246154.0 | 395397.5 | 3.9% | ok |
| small_blocks_gemm_f64_batched_n8_count256 | #1885 | batched | tenferro-cuda | f64 | n=8, count=256 | 1 | 128 | 22493.1 | 123.6 | 0.5% | ok |
| small_blocks_gemm_f64_loop_n16_count256 | #1885 | loop | tenferro-cuda | f64 | n=16, count=256 | 1 | 1 | 7385229.0 | 23396.5 | 2.1% | ok |
| small_blocks_gemm_f64_batched_n16_count256 | #1885 | batched | tenferro-cuda | f64 | n=16, count=256 | 1 | 128 | 20033.8 | 448.8 | 1.9% | ok |
| small_blocks_gemm_f64_loop_n32_count256 | #1885 | loop | tenferro-cuda | f64 | n=32, count=256 | 1 | 1 | 7181693.0 | 284683.5 | 2.8% | ok |
| small_blocks_gemm_f64_batched_n32_count256 | #1885 | batched | tenferro-cuda | f64 | n=32, count=256 | 1 | 128 | 20923.6 | 277.6 | 3.8% | ok |
| small_blocks_gemm_f64_loop_n64_count256 | #1885 | loop | tenferro-cuda | f64 | n=64, count=256 | 1 | 1 | 6946685.0 | 43686.5 | 2.0% | ok |
| small_blocks_gemm_f64_batched_n64_count256 | #1885 | batched | tenferro-cuda | f64 | n=64, count=256 | 1 | 128 | 21719.4 | 238.6 | 0.9% | ok |
| batched_qr_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 5731855.0 | 27866.0 | 2.0% | ok |
| batched_qr_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=16 | 1 | 1 | 2523872.0 | 7735.5 | 0.2% | ok |
| batched_qr_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 112252159.0 | 3099070.5 | 3.1% | ok |
| batched_qr_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=qr, n=32, batch=256 | 1 | 1 | 35994484.0 | 63926.5 | 0.4% | ok |
| batched_svd_f64_loop_n32_b16 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 17109863.0 | 302644.5 | 1.7% | ok |
| batched_svd_f64_batched_n32_b16 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=16 | 1 | 1 | 14347793.0 | 25885.5 | 0.1% | ok |
| batched_svd_f64_loop_n32_b256 | #1885 | loop | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 298362725.0 | 812785.5 | 0.3% | ok |
| batched_svd_f64_batched_n32_b256 | #1885 | batched | tenferro-cuda | f64 | op=svd, n=32, batch=256 | 1 | 1 | 226595634.0 | 341401.0 | 0.7% | ok |
| qr_live_buffers_f64_n32_live16 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=16 | 1 | 8 | 337798.4 | 11932.2 | 3.3% | ok |
| qr_live_buffers_f64_n32_live256 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=256 | 1 | 8 | 383626.1 | 1559.9 | 0.3% | ok |
| qr_live_buffers_f64_n32_live1024 | #1885 | loop | tenferro-cuda | f64 | op=qr, n=32, batch=1, live_buffers=1024 | 1 | 8 | 491573.4 | 1834.4 | 0.4% | ok |

## First-call diagnostics (not steady-state rows)

One cold process; each distinct layout is uploaded outside timing, then its first and second call are timed separately (each includes a device synchronize).

### first_call_transpose_to-contiguous_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.059 ms, second call 0.033 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 1701.288 | 0.084 |
| [2, 5] -> transpose |  | 0.071 | 0.041 |
| [2, 7] -> transpose |  | 0.061 | 0.033 |
| [3, 5] -> transpose |  | 0.060 | 0.032 |
| [3, 7] -> transpose |  | 0.059 | 0.032 |
| [4, 5] -> transpose |  | 0.061 | 0.033 |
| [4, 7] -> transpose |  | 0.062 | 0.032 |
| [5, 5] -> transpose |  | 0.068 | 0.032 |
| [5, 7] -> transpose |  | 0.058 | 0.033 |
| [6, 5] -> transpose |  | 0.059 | 0.033 |
| [6, 7] -> transpose |  | 0.058 | 0.033 |
| [7, 5] -> transpose |  | 0.059 | 0.032 |
| [7, 7] -> transpose |  | 0.059 | 0.032 |
| [8, 5] -> transpose |  | 0.062 | 0.033 |
| [8, 7] -> transpose |  | 0.063 | 0.032 |
| [9, 5] -> transpose |  | 0.055 | 0.032 |
| [9, 7] -> transpose |  | 0.066 | 0.032 |
| [10, 5] -> transpose |  | 0.059 | 0.032 |
| [10, 7] -> transpose |  | 0.058 | 0.033 |
| [11, 5] -> transpose |  | 0.063 | 0.033 |
| [11, 7] -> transpose |  | 0.058 | 0.033 |
| [12, 5] -> transpose |  | 0.058 | 0.032 |
| [12, 7] -> transpose |  | 0.057 | 0.033 |

### first_call_transpose_copy-into_f64_layouts_r2-12_c5-7 (#1885)

Median over 22 distinct layouts: first call 0.039 ms, second call 0.033 ms.

| Layout | Warm-up layout | First call ms | Second call ms |
|---|---|---:|---:|
| [13, 3] -> transpose | yes | 62.325 | 0.057 |
| [2, 5] -> transpose |  | 0.046 | 0.038 |
| [2, 7] -> transpose |  | 0.041 | 0.033 |
| [3, 5] -> transpose |  | 0.037 | 0.033 |
| [3, 7] -> transpose |  | 0.045 | 0.035 |
| [4, 5] -> transpose |  | 0.043 | 0.051 |
| [4, 7] -> transpose |  | 0.042 | 0.037 |
| [5, 5] -> transpose |  | 0.038 | 0.034 |
| [5, 7] -> transpose |  | 0.039 | 0.034 |
| [6, 5] -> transpose |  | 0.038 | 0.033 |
| [6, 7] -> transpose |  | 0.040 | 0.034 |
| [7, 5] -> transpose |  | 0.039 | 0.034 |
| [7, 7] -> transpose |  | 0.048 | 0.036 |
| [8, 5] -> transpose |  | 0.039 | 0.033 |
| [8, 7] -> transpose |  | 0.040 | 0.035 |
| [9, 5] -> transpose |  | 0.042 | 0.032 |
| [9, 7] -> transpose |  | 0.026 | 0.024 |
| [10, 5] -> transpose |  | 0.025 | 0.023 |
| [10, 7] -> transpose |  | 0.025 | 0.023 |
| [11, 5] -> transpose |  | 0.025 | 0.023 |
| [11, 7] -> transpose |  | 0.026 | 0.022 |
| [12, 5] -> transpose |  | 0.024 | 0.022 |
| [12, 7] -> transpose |  | 0.025 | 0.022 |


## Timing boundaries

- **alloc_zero / alloc-fill-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_output + fill_zero_write (device fill), device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-per-callback** — inside: with_cubecl entry/exit per allocation, alloc_zero_output, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / alloc-zero-within-callback** — inside: one with_cubecl entry/exit per batch, alloc_zero_output x iterations, device synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **alloc_zero / host-zero-upload** — inside: upload_tensor of a host zero buffer, runtime synchronize at batch end; outside: backend construction, backend session entry/exit, host zero buffer construction, zero check (download).
- **batched_factorization / batched** — inside: one batched factorization call ([n, n, batch]), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **batched_factorization / loop** — inside: batch x rank-2 factorization (per-matrix submissions and info checks), device synchronize at batch end; outside: backend construction, uploads (and live resident buffers), session entry/exit, reconstruction check (download).
- **first_call_layouts / copy-into** — inside: to_contiguous (or copy_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **first_call_layouts / to-contiguous** — inside: to_contiguous (or copy_into a compact destination) of a transposed device view + device synchronize, first and second call per distinct layout; outside: backend construction, upload, warm-up layout, download check.
- **small_blocks / batched** — inside: one batched dot_general_read ([n, n, count]), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **small_blocks / loop** — inside: count x dot_general_read (one submission per block), device synchronize at batch end; outside: backend construction, uploads, session entry/exit, correctness check (download).
- **transfer / cudarc-pageable** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / cudarc-pinned** — inside: cudarc memcpy into a preallocated buffer, stream.synchronize per call; outside: backend/context construction, host and device buffer allocation, round-trip check.
- **transfer / tenferro** — inside: download_tensor (synchronizes itself), host output allocation; outside: backend/context construction, host and device buffer allocation, round-trip check.

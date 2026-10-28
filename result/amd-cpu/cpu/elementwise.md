# CPU Elementwise Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/elementwise`
- Suite file: `benchmarks/cpu/elementwise.yaml`
- Timestamp: `2026-10-28T03:20:54.546057+00:00`
- tenferro-rs commit: `e80de9ac63675b19dbe4137e7de0c788283dd8c0`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `unknown`
- Cores per socket: `64`
- Threads per core: `unknown`
- NUMA nodes: `unknown`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-6.8.0-147-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-7; 8-15; 16-23; 24-31; 32-39; 40-47; 48-55; 56-63`
- CPU boost: `enabled`

f64 `sqrt` / `rsqrt` / `exp` on 33_554_432 elements through the tenferro-rs eager CPU API. Each iteration drops the output tensor and only black-boxes its shape and dtype, so nothing retains it: under the default 100 MiB buffer-pool limit a 256 MiB output cannot come back through the pool, so the measured call pays for a new destination (allocation and, typically, first touch), while the raised 4 GiB limit lets the next call reuse the previous output buffer. There is deliberately no external reference backend: JAX and PyTorch differ in allocator and caching strategy, which is the difference this suite isolates. The huge-page mode above is part of the reading: with `madvise` a fresh large destination faults 4 KiB pages, with `always` the kernel may hand out 2 MiB pages and the default arm gets cheaper. See `docs/elementwise-suite.md` and tensor4all/tenferro-rs#2284.

Thread counts come from the engine: a one-thread run uses `CpuEngine::serial()`, a wider run gets its own pool, and every row records the width the engine reports. `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` are also set so any library-level parallelism matches; no CPU-affinity pinning (`taskset` / `numactl`) is applied by the scripts, matching the other CPU suites. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`default` is `DEFAULT_MAX_RETAINED_CAPACITY_BYTES` (100 MiB), set explicitly on the engine so an inherited `TENFERRO_BUFFER_POOL_MAX_RETAINED_BYTES` cannot move the baseline; `raised` is the same engine with `set_buffer_pool_limit_bytes(4 GiB)`. `IQR` is `p75 - p25`, `GB/s` is logical bytes (`2 * n * 8`) over the median, and `default/raised` is the **pool-policy slowdown**: the timed call covers the operation and the destination it has to take (allocation and first touch when the pool cannot help, a pool hand-back when it can), with recycling itself after the clock stops. The investigation in tensor4all/tenferro-rs#2284 attributes the dominant term to destination first touch.

## Threads: 1

| op | pool | median ms | IQR ms | GB/s | default/raised |
|---|---|---:|---:|---:|---:|
| `sqrt` | default (100 MiB) | 141.208 | 20.600 | 3.80 | - |
| `sqrt` | raised (4 GiB) | 21.709 | 0.403 | 24.73 | 6.50x |
| `rsqrt` | default (100 MiB) | 144.567 | 0.897 | 3.71 | - |
| `rsqrt` | raised (4 GiB) | 30.812 | 0.453 | 17.42 | 4.69x |
| `exp` | default (100 MiB) | 185.831 | 2.289 | 2.89 | - |
| `exp` | raised (4 GiB) | 72.290 | 0.961 | 7.43 | 2.57x |

## Threads: 4

| op | pool | median ms | IQR ms | GB/s | default/raised |
|---|---|---:|---:|---:|---:|
| `sqrt` | default (100 MiB) | 39.287 | 3.221 | 13.66 | - |
| `sqrt` | raised (4 GiB) | 14.286 | 0.296 | 37.58 | 2.75x |
| `rsqrt` | default (100 MiB) | 43.041 | 0.506 | 12.47 | - |
| `rsqrt` | raised (4 GiB) | 14.051 | 0.159 | 38.21 | 3.06x |
| `exp` | default (100 MiB) | 54.375 | 4.176 | 9.87 | - |
| `exp` | raised (4 GiB) | 19.782 | 2.411 | 27.14 | 2.75x |

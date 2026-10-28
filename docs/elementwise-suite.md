# CPU elementwise suite (`cpu/elementwise`)

Standalone suite, like `cpu/permutation`: not part of `scripts/run_all.sh`'s
default path.

## Why it exists

tensor4all/tenferro-rs#2284 reported f64 `sqrt` / `rsqrt` on 33_554_432
elements as 2.3-3.4x slower than JAX and PyTorch at 1 thread. The kernel was not
the cause. The CPU buffer pool retains at most
`DEFAULT_MAX_RETAINED_CAPACITY_BYTES` (100 MiB), a 256 MiB output is larger than
that, so the output is never retained and every eager call first-touches a fresh
allocation. Raising the limit with `CpuEngine::set_buffer_pool_limit_bytes` took
1-thread `sqrt` from 133 ms to 24 ms on the measuring host.

## The two pool configurations

Every op is measured twice, with a fresh engine per configuration:

| pool | engine state |
|---|---|
| `default` | `set_buffer_pool_limit_bytes(100 MiB)`, the shipped `DEFAULT_MAX_RETAINED_CAPACITY_BYTES`, pinned explicitly |
| `raised` | `set_buffer_pool_limit_bytes(1 << 32)` (4 GiB), before timing |

The output tensor is dropped after every iteration (and after the clock stops)
and only its shape and dtype are black-boxed, so nothing keeps it alive. With
`default` the 256 MiB output cannot come back through the pool, so each measured
call pays for a new destination - allocation and, typically, first touch. With
`raised` the pool hands the same buffer back. The pool only returns storage to
the process allocator, so page reuse below that layer is the allocator's
business; the report prices the policy, not page-fault counts.

**The `default`/`raised` ratio is the pool-policy slowdown.** The timed call
covers the operation plus the destination it has to take: allocation and first
touch when the pool cannot help, a pool hand-back when it can. Recycling itself
happens after the clock stops, and the buffer is back in the pool before the next
sample. The ratio prices the retention policy as a whole rather than page-fault
counts (the #2284 investigation attributes the dominant term to first touch). It is not a kernel
speedup and should not be read as one. A ratio near 1 means the op is not
limited by output allocation on this host.

## No external reference

The suite deliberately has no JAX/PyTorch/HPTT/Julia participant. Those
implementations differ in allocator and caching strategy, and that difference is
exactly what produced the original 3.4x gap, so an external cell would compare
memory management rather than kernels. The pool configurations are the
comparison.

## How to run it

```bash
BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_elementwise.sh 1 4
```

Thread counts are looped over sequentially, in a fresh process per count. The
engine owns the workers (one thread is `CpuEngine::serial()`, wider runs get a
private pool) and each row records the width the engine reports;
`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` are set as well so any library-level
parallelism matches. Raw runs land in
`data/results/<target_profile>/cpu/elementwise/<timestamp>/`
(`run.yaml`, `run_t<N>.yaml`, `elementwise_t<N>_<timestamp>.csv`) and the report
is written to `result/<target_profile>/cpu/elementwise.md`.

The elementwise path uses no BLAS, so the script defaults to
`TENFERRO_CPU_FEATURES=native` (tenferro's own kernels) and needs no
OpenBLAS/MKL installation; override the variable to measure another provider.

## Suite file

`benchmarks/cpu/elementwise.yaml` lists the three ops and validates against
`schemas/benchmark-suite.schema.json`. That schema has no problem field for an
engine/buffer-pool configuration, so the two configurations are not separate
problems: they are the `pool_limit_bytes` column of the runner's CSV rows and
the `pool` column of the report table.

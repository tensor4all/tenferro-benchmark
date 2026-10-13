# CUDA shape-stream diagnostics

The `shape_stream` cases in `gpu/perf_issues` address the GPU shape-sequence
requirements of benchmark issue #107 and the plan/workspace workloads of
tenferro-rs #1873 and #1805. They complement the existing diagnostic that times
only the final contraction after cache churn.

Each latency sample is the entire ordered sequence, followed by native CUDA
synchronization. Required contraction planning is included. Inputs, public read
handles, output-retention storage, one entered session and explicit library
initialization are prepared outside timing, including with zero warmups.
Every output survives the clock stop; full-value validation and destruction
occur afterward. The raw row records enqueue and synchronized sequence durations,
exact keys, shapes and strides. These are workflow latency screens, not
steady-state throughput or confirmed baseline/candidate performance comparisons.

Before each sample, clear CUDA extension-owned plans/workspace and drain the
device. Keep compiled CubeCL kernels, the cuTENSOR library handle and allocator
pools initialized. Set the contraction-plan capacity to 64. The modes are:

| Mode | Sequence | Precondition |
| --- | --- | --- |
| `fixed-warm` | 32 calls to one key | Warm the key after reset |
| `mixed-warm` | 32 ordered revisits of three keys | Warm all three after reset |
| `fresh` | Eight distinct keys | Start with empty contraction cache |
| `capacity-thrash` | 72 distinct keys, then revisit key zero | Start empty; exceed the 64-plan bound |
| `equal-shape-strides` | 32 alternating compact/transposed borrowed reads | Same logical shape/values; warm both layouts |

All modes use F64 batched contractions with base sizes 16 and 64. The versioned
quick selection contains fixed/mixed/fresh latency cases; full adds capacity and
stride cases and every counter pass. Coverage and sampling effort remain separate.
Counter cases run the same declared precondition and sequence with no clocks,
observing public plan hits/misses/evictions/bytes, workspace bytes and retirement
state after enqueue, completion and output drop. Counters are cumulative; compare
snapshots within the pass. Raw timing rows contain no counter queries.

Run builds and checks in the repository CUDA devcontainer. Physical GPU1 is for
correctness/counter work; GPU0 is for sequential timing collection. For example:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 BENCH_COVERAGE=full BENCH_INSTANCE=shape_stream_f64_n16_fresh_latency,shape_stream_f64_n16_fresh_counters PERF_ISSUES_CORRECTNESS_ONLY=1 ./scripts/run_gpu_perf_issues.sh'
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_COVERAGE=quick BENCH_EFFORT=scan ./scripts/run_gpu_perf_issues.sh'
```

The runner rebuilds before execution. Correctness-only shape-stream rows take no
clocks. A filtered run records unselected cases and cannot establish complete
coverage. Reports stay in `result/nvidia-gpu/gpu/perf_issues.md`; raw repetitions,
selection accounting and metadata stay below `data/results/nvidia-gpu/gpu/perf_issues/`.
Use the paired confirmation workflow before making a regression or speedup claim.

The dedicated [shape-stream report](../result/nvidia-gpu/gpu/shape_stream.md)
combines GPU0 sequence latency and a separate GPU1 untimed counter pass. Its
formatter verifies case completeness within this family, numeric checks, matching
operation source/sequence, GPU roles, and absence of counters in timed rows or
timings in counter rows. It preserves the selected raw runs' revisions and effort.
Regenerate from recorded runs, for example:

```bash
uv run python scripts/format_gpu_shape_stream_results.py \
  --timing-run data/results/nvidia-gpu/gpu/perf_issues/20261013_221324 \
  --counter-run data/verification/nvidia-gpu/gpu/shape_stream/20261013_215852 \
  --output result/nvidia-gpu/gpu/shape_stream.md
```

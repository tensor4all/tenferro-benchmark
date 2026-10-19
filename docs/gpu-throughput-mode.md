# GPU Throughput Mode (Deferred Batch)

Throughput mode is a **separate measurement mode** from the per-call latency
suites. It uses the deferred prepared API added for tenferro-rs #2087:

```rust
session.with_session(|view| {
    let mut deferred = runtime.begin_prepared_deferred(view, &prepared)?;
    for _ in 0..k {
        deferred.run_prepared_deferred(&inputs)?;
    }
    deferred.finish_prepared()          // one retirement for the whole batch
})
```

A latency row pays one retirement barrier per call. A throughput row enqueues
`K` prepared runs in one caller-held CUDA session and pays **one** retirement
per batch, so its reported per-operation time is `sample_ms / K`.

## Where the two modes are kept apart

| | latency | throughput |
|---|---|---|
| suite files | `benchmarks/gpu/dense.yaml`, `elementwise.yaml`, `einsum.yaml`, `linalg_jvp_vjp.yaml`, `linalg_ad_latency.yaml` | `benchmarks/gpu/dense_throughput.yaml`, `benchmarks/gpu/linalg_ad_throughput.yaml` |
| suite ids | `gpu/dense`, `gpu/linalg_ad_latency`, ... | `gpu/dense_throughput`, `gpu/linalg_ad_throughput` |
| JSONL `backend` | `tenferro-cuda-trace` | `tenferro-cuda-trace-deferred-batch` |
| JSONL `execution.execution_path` | `trace-run-prepared-in-held-session`, `trace-run-prepared-in-held-session-linalg-ad`, ... | `...-deferred-batch` |
| JSONL notes | per-call setup note | `mode: throughput`, `execution_mode: deferred-batch`, `deferred_batch.{configured_k,effective_k,retirements_per_sample}` |
| reports | `result/nvidia-gpu/gpu/dense.md`, ... | `result/nvidia-gpu/gpu/dense_throughput.md`, `result/nvidia-gpu/gpu/linalg_ad_throughput.md` |

Because the backend identity differs, the formatters and the dashboard summary
sidecars (`*.summary.json`) keep the two modes in separate rows. Never put a
latency and a throughput number in one row, ratio, average or summary: a
throughput row is a batched measurement and is not a single-call cost.

The suite also declares the regime in the YAML (`defaults.regime: throughput`
and `regime: throughput` on each problem), per the repository regime rule. The
AD runner selects the deferred route from the problem's `regime`; the dense
runner selects it from the backend name (`tenferro-cuda-trace-deferred-batch`).

## Batch size

`K` comes from `GPU_BENCH_DEFERRED_BATCH_K` (default `64`). It is bounded by the
output-retention budget: `effective_k = min(K, 256 MiB / one output set)`, and
both values are recorded in the row's notes. The guard is entered once per
sampling run, outside the clocks; a timed sample contains only the `K`
`run_prepared_deferred` enqueues and the single `finish_prepared` whose
retirement barrier waits for the device. Verification downloads the batch
outputs after the clock stops.

## Eligibility

`Runtime::begin_prepared_deferred` rejects a graph that is not one
borrowed-session engine. A rejected case is reported as `unsupported` with the
preflight message; the harness never falls back to the per-call latency route,
because that would silently mix the two regimes.

## Collection

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_throughput.sh'

# K override
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEFERRED_BATCH_K=16 \
    ./scripts/run_gpu_throughput.sh'
```

Raw records land in
`data/results/nvidia-gpu/gpu/{dense_throughput,linalg_ad_throughput}/<timestamp>/`
and the reports in `result/nvidia-gpu/gpu/{dense_throughput,linalg_ad_throughput}.md`.
The `run.yaml` of each run records `collection.mode = "throughput"` and the
batch `K`. Run the collection sequentially; do not overlap it with another GPU
benchmark process.

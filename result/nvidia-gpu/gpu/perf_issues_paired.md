- Raw campaign: `data/results/nvidia-gpu/gpu/perf_issues_paired/20261014_001820`.

- Command: `scripts/benchmark_gpu_paired.py confirm --baseline /workspaces/tenferro-benchmark/extern/tenferro-confirm-baseline --candidate /workspaces/tenferro-benchmark/extern/tenferro-confirm-baseline --config /tmp/gpu-confirmation.yaml --aa-dir data/results/nvidia-gpu/gpu/perf_issues_aa/20261014_000326`.

Focused declared subset only; globally unselected suite cases remain unmeasured. One declared unit is one allocation or one complete controlled shape sequence. Raw samples, sequence/cache contracts and build/hardware metadata remain in every arm.

# Regression detection (confirm)

- Overall: **OK**

## Deterministic findings

None.

## Timing

| Case | Verdict | Statistic | Rounds | A/A spread | Reason |
|---|---|---:|---:|---:|---|
| `alloc_zero_f64_alloc-zero-per-callback_16el@t1` | NO_CHANGE | 0.981 | 4 | 0.005 | within the declared thresholds |
| `alloc_zero_f64_alloc-zero-per-callback_32768el@t1` | NO_CHANGE | 0.940 | 4 | 0.038 | within the declared thresholds |
| `alloc_zero_f64_alloc-zero-within-callback_16el@t1` | NO_CHANGE | 1.108 | 4 | 0.004 | within the declared thresholds |
| `alloc_zero_f64_alloc-zero-within-callback_32768el@t1` | NO_CHANGE | 0.964 | 4 | 0.056 | within the declared thresholds |
| `shape_stream_f64_n16_fixed-warm_latency@t1` | NO_CHANGE | 0.998 | 4 | 0.059 | within the declared thresholds |
| `shape_stream_f64_n16_fresh_latency@t1` | NO_CHANGE | 1.010 | 4 | 0.011 | within the declared thresholds |
| `shape_stream_f64_n16_mixed-warm_latency@t1` | NO_CHANGE | 0.986 | 4 | 0.014 | within the declared thresholds |
| `shape_stream_f64_n64_fixed-warm_latency@t1` | NO_CHANGE | 1.070 | 4 | 0.021 | within the declared thresholds |
| `shape_stream_f64_n64_fresh_latency@t1` | NO_CHANGE | 0.965 | 4 | 0.007 | within the declared thresholds |
| `shape_stream_f64_n64_mixed-warm_latency@t1` | NO_CHANGE | 0.984 | 4 | 0.021 | within the declared thresholds |

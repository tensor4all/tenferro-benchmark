- Raw campaign: `data/results/nvidia-gpu/gpu/perf_issues_aa/20261014_000326`.

- Command: `scripts/benchmark_gpu_paired.py aa --baseline /workspaces/tenferro-benchmark/extern/tenferro-confirm-baseline --rounds 4 --warmups 3 --runs 15`.

Focused declared subset only; globally unselected suite cases remain unmeasured. One declared unit is one allocation or one complete controlled shape sequence. Raw samples, sequence/cache contracts and build/hardware metadata remain in every arm.

# A/A noise characterization

Same tenferro-rs build in both arms, balanced order. Use these spreads to declare `noise.max_aa_relative_spread`, `noise.max_cov` and the relative threshold in the confirmation config **before** any candidate run. No verdicts are produced here.

- Balanced order: yes

| Case | A/A statistic | Spread | Max CoV | Rounds | Round ratios |
|---|---:|---:|---:|---:|---|
| `alloc_zero_f64_alloc-zero-per-callback_16el@t1` | 1.0051 | 0.0051 | 0.0708 | 4 | 0.9252, 1.0946, 1.0347, 0.9754 |
| `alloc_zero_f64_alloc-zero-per-callback_32768el@t1` | 0.9616 | 0.0384 | 0.0681 | 4 | 0.9433, 0.9799, 0.9140, 1.0039 |
| `alloc_zero_f64_alloc-zero-within-callback_16el@t1` | 1.0038 | 0.0038 | 0.0648 | 4 | 1.1682, 1.0414, 0.8746, 0.9662 |
| `alloc_zero_f64_alloc-zero-within-callback_32768el@t1` | 1.0557 | 0.0557 | 0.0706 | 4 | 1.0979, 0.9997, 1.0738, 1.0377 |
| `shape_stream_f64_n16_fixed-warm_latency@t1` | 0.9414 | 0.0586 | 0.0565 | 4 | 0.9553, 0.9274, 0.9000, 0.9907 |
| `shape_stream_f64_n16_fresh_latency@t1` | 1.0108 | 0.0108 | 0.1008 | 4 | 0.9526, 1.0011, 1.1316, 1.0205 |
| `shape_stream_f64_n16_mixed-warm_latency@t1` | 0.9856 | 0.0144 | 0.0649 | 4 | 1.0053, 0.9823, 0.9020, 0.9889 |
| `shape_stream_f64_n64_fixed-warm_latency@t1` | 0.9787 | 0.0213 | 0.0697 | 4 | 1.1164, 0.9635, 0.9938, 0.9578 |
| `shape_stream_f64_n64_fresh_latency@t1` | 0.9925 | 0.0075 | 0.0941 | 4 | 1.0163, 0.9317, 0.9704, 1.0146 |
| `shape_stream_f64_n64_mixed-warm_latency@t1` | 1.0207 | 0.0207 | 0.0614 | 4 | 1.0199, 1.0582, 0.9684, 1.0215 |

# Public CUDA lifetime counters

Raw observations: `data/verification/nvidia-gpu/gpu/compare_recheck/20261013_192206/lifetime_counters.jsonl`.

These observations were collected separately from timing on debug GPU1. They are not performance measurements or standard operation coverage. Each case uses one entered backend session and retains outputs until native completion.

Driver free memory covers the whole device. Its change cannot isolate an output pool, prove a leak, or explain the separately observed latency growth. No public output-pool statistics or clear API is available.

| Case | Batches | Operations/batch | Maximum normalized error | Driver free-byte change after final drop | Workspace retained bytes | Retirement in flight / leaked |
|---|---:|---:|---:|---:|---:|---:|
| compare_f32_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |
| compare_f64_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |
| compare_eq_f32_n1024 | 30 | 128 | 0 | -8388608 | 0 | 0 / 0 |
| compare_lt_f32_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |
| compare_le_f32_n1024 | 30 | 128 | 0 | -8388608 | 0 | 0 / 0 |
| compare_ge_f32_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |
| compare_eq_f64_n1024 | 30 | 128 | 0 | -8388608 | 0 | 0 / 0 |
| compare_lt_f64_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |
| compare_le_f64_n1024 | 30 | 128 | 0 | -8388608 | 0 | 0 / 0 |
| compare_ge_f64_n1024 | 30 | 128 | 0 | 0 | 0 | 0 / 0 |

Raw JSONL preserves the cache entries, bytes, hits, misses, evictions, clears, workspace retirement counters, and pending transfer retirements at enqueue, completion, and output drop. Queries can perturb execution; they are not synchronization primitives.

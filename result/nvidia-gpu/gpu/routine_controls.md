# GPU routine latency and state controls

Source: `/workspaces/tenferro-benchmark/.worktrees/gpu-lane/data/results/nvidia-gpu/gpu/routine_controls/20261013_234113`.

Manifest 1; coverage quick; effort standard.

Timing controls measured and state controls observed: yes.

State/counter controls contain no timed samples. Verification-only and unselected cases are not measured coverage. These controls do not establish a paired regression conclusion.

Revision/features mismatch across lanes: False.

Raw run: `/workspaces/tenferro-benchmark/.worktrees/gpu-lane/data/results/nvidia-gpu/gpu/perf_issues/20261013_234113_timing`; tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`; GPU UUID `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`.

Raw run: `/workspaces/tenferro-benchmark/.worktrees/gpu-lane/data/results/nvidia-gpu/gpu/perf_issues/20261013_234113_state`; tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`; GPU UUID `GPU-ed896702-ca5c-93f6-2537-06fffdac4a91`.

| Classification | Count |
|---|---:|
| timing | 10 |
| state | 8 |
| unselected | 24 |
| expected | 18 |
| selected | 18 |
| measured | 10 |
| observed | 8 |
| verification_only | 0 |
| missing | 0 |
| failed | 0 |
| noisy | 0 |
| duplicates | 0 |
| unexpected | 0 |

## Latency observations

Times are per declared unit, including host dispatch and native completion synchronization. Whole-sequence samples include required planning; shared allocation and per-call session entry are distinct scopes. Host enqueue is not device kernel time. Setup and validation are untimed.

| Case | Scope | Unit | Calls/sequence | Samples | Median µs | IQR µs | Host enqueue µs |
|---|---|---|---:|---:|---:|---:|---:|
| shape_stream_f64_n16_fixed-warm_latency | shape_stream_diagnostic | sequence | 32 | 7 | 785.383 | 70.959 | 776.767 |
| shape_stream_f64_n16_mixed-warm_latency | shape_stream_diagnostic | sequence | 32 | 7 | 813.807 | 12.607 | 803.876 |
| shape_stream_f64_n16_fresh_latency | shape_stream_diagnostic | sequence | 8 | 7 | 520.794 | 18.157 | 508.945 |
| shape_stream_f64_n64_fixed-warm_latency | shape_stream_diagnostic | sequence | 32 | 7 | 817.924 | 19.279 | 809.364 |
| shape_stream_f64_n64_mixed-warm_latency | shape_stream_diagnostic | sequence | 32 | 7 | 869.455 | 57.114 | 861.072 |
| shape_stream_f64_n64_fresh_latency | shape_stream_diagnostic | sequence | 8 | 7 | 526.355 | 14.366 | 515.889 |
| alloc_zero_f64_alloc-zero-within-callback_16el | steady_state | allocation | — | 7 | 8.924 | 0.767 | — |
| alloc_zero_f64_alloc-zero-per-callback_16el | session_entry_diagnostic | allocation | — | 7 | 18.897 | 0.425 | — |
| alloc_zero_f64_alloc-zero-within-callback_32768el | steady_state | allocation | — | 7 | 11.370 | 0.645 | — |
| alloc_zero_f64_alloc-zero-per-callback_32768el | session_entry_diagnostic | allocation | — | 7 | 21.055 | 1.672 | — |

## Untimed state observations

Plan bytes and workspace describe public extension retention, not the whole allocator. A separate plan-build count is unavailable; misses are not substituted for builds. Observed in-flight peak uses snapshots and can miss activity between them. RSS/free-memory changes alone do not establish a leak.

| Case | Public observation |
|---|---|
| shape_stream_f64_n16_fixed-warm_counters | `{"entries_after": 1, "entries_before": 1, "evictions": 0, "hits": 32, "misses": 0, "observed_in_flight_peak": 0, "plan_bytes": 884, "workspace_bytes": 16777216}` |
| shape_stream_f64_n16_mixed-warm_counters | `{"entries_after": 3, "entries_before": 3, "evictions": 0, "hits": 32, "misses": 0, "observed_in_flight_peak": 0, "plan_bytes": 2412, "workspace_bytes": 16777216}` |
| shape_stream_f64_n16_fresh_counters | `{"entries_after": 8, "entries_before": 0, "evictions": 0, "hits": 0, "misses": 8, "observed_in_flight_peak": 0, "plan_bytes": 6232, "workspace_bytes": 16777216}` |
| shape_stream_f64_n64_fixed-warm_counters | `{"entries_after": 1, "entries_before": 1, "evictions": 0, "hits": 32, "misses": 0, "observed_in_flight_peak": 0, "plan_bytes": 884, "workspace_bytes": 16777216}` |
| shape_stream_f64_n64_mixed-warm_counters | `{"entries_after": 3, "entries_before": 3, "evictions": 0, "hits": 32, "misses": 0, "observed_in_flight_peak": 0, "plan_bytes": 2412, "workspace_bytes": 16777216}` |
| shape_stream_f64_n64_fresh_counters | `{"entries_after": 8, "entries_before": 0, "evictions": 0, "hits": 0, "misses": 8, "observed_in_flight_peak": 0, "plan_bytes": 6232, "workspace_bytes": 16777216}` |
| vendor_churn_f64_cycles8 | `{"cycles": 8, "free_bytes_after": 11825446912, "free_bytes_before": 11844321280, "rss_bytes_after": 729649152, "rss_bytes_before": 729075712, "rss_growth_bytes": 573440}` |
| vendor_churn_f64_cycles64 | `{"cycles": 64, "free_bytes_after": 11768823808, "free_bytes_before": 11844321280, "rss_bytes_after": 728559616, "rss_bytes_before": 727908352, "rss_growth_bytes": 651264}` |

## Sequence/cache contracts

Key order is exact. Fixture definitions retain shapes and strides; full per-command enqueue/completion/drop snapshots remain in the referenced raw runs.

- `shape_stream_f64_n16_fixed-warm_latency`: mode `fixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n16_mixed-warm_latency`: mode `mixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [17, 2, 16], "lhs_strides": [1, 17, 34], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [18, 2, 16], "lhs_strides": [1, 18, 36], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n16_fresh_latency`: mode `fresh`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 3, 4, 5, 6, 7]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [17, 2, 16], "lhs_strides": [1, 17, 34], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [18, 2, 16], "lhs_strides": [1, 18, 36], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 3, "lhs_layout": "compact", "lhs_shape": [19, 2, 16], "lhs_strides": [1, 19, 38], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 4, "lhs_layout": "compact", "lhs_shape": [20, 2, 16], "lhs_strides": [1, 20, 40], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 5, "lhs_layout": "compact", "lhs_shape": [21, 2, 16], "lhs_strides": [1, 21, 42], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 6, "lhs_layout": "compact", "lhs_shape": [22, 2, 16], "lhs_strides": [1, 22, 44], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 7, "lhs_layout": "compact", "lhs_shape": [23, 2, 16], "lhs_strides": [1, 23, 46], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n64_fixed-warm_latency`: mode `fixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.
- `shape_stream_f64_n64_mixed-warm_latency`: mode `mixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [65, 2, 64], "lhs_strides": [1, 65, 130], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [66, 2, 64], "lhs_strides": [1, 66, 132], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.
- `shape_stream_f64_n64_fresh_latency`: mode `fresh`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 3, 4, 5, 6, 7]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [65, 2, 64], "lhs_strides": [1, 65, 130], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [66, 2, 64], "lhs_strides": [1, 66, 132], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 3, "lhs_layout": "compact", "lhs_shape": [67, 2, 64], "lhs_strides": [1, 67, 134], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 4, "lhs_layout": "compact", "lhs_shape": [68, 2, 64], "lhs_strides": [1, 68, 136], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 5, "lhs_layout": "compact", "lhs_shape": [69, 2, 64], "lhs_strides": [1, 69, 138], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 6, "lhs_layout": "compact", "lhs_shape": [70, 2, 64], "lhs_strides": [1, 70, 140], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 7, "lhs_layout": "compact", "lhs_shape": [71, 2, 64], "lhs_strides": [1, 71, 142], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.
- `shape_stream_f64_n16_fixed-warm_counters`: mode `fixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n16_mixed-warm_counters`: mode `mixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [17, 2, 16], "lhs_strides": [1, 17, 34], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [18, 2, 16], "lhs_strides": [1, 18, 36], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n16_fresh_counters`: mode `fresh`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 3, 4, 5, 6, 7]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [16, 2, 16], "lhs_strides": [1, 16, 32], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [17, 2, 16], "lhs_strides": [1, 17, 34], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [18, 2, 16], "lhs_strides": [1, 18, 36], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 3, "lhs_layout": "compact", "lhs_shape": [19, 2, 16], "lhs_strides": [1, 19, 38], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 4, "lhs_layout": "compact", "lhs_shape": [20, 2, 16], "lhs_strides": [1, 20, 40], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 5, "lhs_layout": "compact", "lhs_shape": [21, 2, 16], "lhs_strides": [1, 21, 42], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 6, "lhs_layout": "compact", "lhs_shape": [22, 2, 16], "lhs_strides": [1, 22, 44], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]} {"key": 7, "lhs_layout": "compact", "lhs_shape": [23, 2, 16], "lhs_strides": [1, 23, 46], "rhs_shape": [16, 2, 3], "rhs_strides": [1, 16, 32]}`.
- `shape_stream_f64_n64_fixed-warm_counters`: mode `fixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.
- `shape_stream_f64_n64_mixed-warm_counters`: mode `mixed-warm`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [65, 2, 64], "lhs_strides": [1, 65, 130], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [66, 2, 64], "lhs_strides": [1, 66, 132], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.
- `shape_stream_f64_n64_fresh_counters`: mode `fresh`; reset `CUDA extension-owned plans/workspace; excludes CubeCL kernels, cuTENSOR handle and output/allocator pools`; order `[0, 1, 2, 3, 4, 5, 6, 7]`; fixtures `{"key": 0, "lhs_layout": "compact", "lhs_shape": [64, 2, 64], "lhs_strides": [1, 64, 128], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 1, "lhs_layout": "compact", "lhs_shape": [65, 2, 64], "lhs_strides": [1, 65, 130], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 2, "lhs_layout": "compact", "lhs_shape": [66, 2, 64], "lhs_strides": [1, 66, 132], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 3, "lhs_layout": "compact", "lhs_shape": [67, 2, 64], "lhs_strides": [1, 67, 134], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 4, "lhs_layout": "compact", "lhs_shape": [68, 2, 64], "lhs_strides": [1, 68, 136], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 5, "lhs_layout": "compact", "lhs_shape": [69, 2, 64], "lhs_strides": [1, 69, 138], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 6, "lhs_layout": "compact", "lhs_shape": [70, 2, 64], "lhs_strides": [1, 70, 140], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]} {"key": 7, "lhs_layout": "compact", "lhs_shape": [71, 2, 64], "lhs_strides": [1, 71, 142], "rhs_shape": [64, 2, 3], "rhs_strides": [1, 64, 128]}`.

## expected

- `shape_stream_f64_n16_fixed-warm_latency`
- `shape_stream_f64_n16_mixed-warm_latency`
- `shape_stream_f64_n16_fresh_latency`
- `shape_stream_f64_n64_fixed-warm_latency`
- `shape_stream_f64_n64_mixed-warm_latency`
- `shape_stream_f64_n64_fresh_latency`
- `alloc_zero_f64_alloc-zero-within-callback_16el`
- `alloc_zero_f64_alloc-zero-per-callback_16el`
- `alloc_zero_f64_alloc-zero-within-callback_32768el`
- `alloc_zero_f64_alloc-zero-per-callback_32768el`
- `shape_stream_f64_n16_fixed-warm_counters`
- `shape_stream_f64_n16_mixed-warm_counters`
- `shape_stream_f64_n16_fresh_counters`
- `shape_stream_f64_n64_fixed-warm_counters`
- `shape_stream_f64_n64_mixed-warm_counters`
- `shape_stream_f64_n64_fresh_counters`
- `vendor_churn_f64_cycles8`
- `vendor_churn_f64_cycles64`

## selected

- `shape_stream_f64_n16_fixed-warm_latency`
- `shape_stream_f64_n16_mixed-warm_latency`
- `shape_stream_f64_n16_fresh_latency`
- `shape_stream_f64_n64_fixed-warm_latency`
- `shape_stream_f64_n64_mixed-warm_latency`
- `shape_stream_f64_n64_fresh_latency`
- `alloc_zero_f64_alloc-zero-within-callback_16el`
- `alloc_zero_f64_alloc-zero-per-callback_16el`
- `alloc_zero_f64_alloc-zero-within-callback_32768el`
- `alloc_zero_f64_alloc-zero-per-callback_32768el`
- `shape_stream_f64_n16_fixed-warm_counters`
- `shape_stream_f64_n16_mixed-warm_counters`
- `shape_stream_f64_n16_fresh_counters`
- `shape_stream_f64_n64_fixed-warm_counters`
- `shape_stream_f64_n64_mixed-warm_counters`
- `shape_stream_f64_n64_fresh_counters`
- `vendor_churn_f64_cycles8`
- `vendor_churn_f64_cycles64`

## measured

- `shape_stream_f64_n16_fixed-warm_latency`
- `shape_stream_f64_n16_mixed-warm_latency`
- `shape_stream_f64_n16_fresh_latency`
- `shape_stream_f64_n64_fixed-warm_latency`
- `shape_stream_f64_n64_mixed-warm_latency`
- `shape_stream_f64_n64_fresh_latency`
- `alloc_zero_f64_alloc-zero-within-callback_16el`
- `alloc_zero_f64_alloc-zero-per-callback_16el`
- `alloc_zero_f64_alloc-zero-within-callback_32768el`
- `alloc_zero_f64_alloc-zero-per-callback_32768el`

## observed

- `shape_stream_f64_n16_fixed-warm_counters`
- `shape_stream_f64_n16_mixed-warm_counters`
- `shape_stream_f64_n16_fresh_counters`
- `shape_stream_f64_n64_fixed-warm_counters`
- `shape_stream_f64_n64_mixed-warm_counters`
- `shape_stream_f64_n64_fresh_counters`
- `vendor_churn_f64_cycles8`
- `vendor_churn_f64_cycles64`

## verification_only

None.

## failed

None.

## missing

None.

## noisy

None.

## unselected

- `shape_stream_f64_n16_capacity-thrash_latency`
- `shape_stream_f64_n16_equal-shape-strides_latency`
- `shape_stream_f64_n64_capacity-thrash_latency`
- `shape_stream_f64_n64_equal-shape-strides_latency`
- `alloc_zero_f64_alloc-zero-within-callback_0el`
- `alloc_zero_f64_alloc-zero-per-callback_0el`
- `alloc_zero_f64_host-zero-upload_0el`
- `alloc_zero_f64_alloc-fill-per-callback_0el`
- `alloc_zero_f64_host-zero-upload_16el`
- `alloc_zero_f64_alloc-fill-per-callback_16el`
- `alloc_zero_f64_alloc-zero-within-callback_512el`
- `alloc_zero_f64_alloc-zero-per-callback_512el`
- `alloc_zero_f64_host-zero-upload_512el`
- `alloc_zero_f64_alloc-fill-per-callback_512el`
- `alloc_zero_f64_host-zero-upload_32768el`
- `alloc_zero_f64_alloc-fill-per-callback_32768el`
- `alloc_zero_f64_alloc-zero-within-callback_2097152el`
- `alloc_zero_f64_alloc-zero-per-callback_2097152el`
- `alloc_zero_f64_host-zero-upload_2097152el`
- `alloc_zero_f64_alloc-fill-per-callback_2097152el`
- `shape_stream_f64_n16_capacity-thrash_counters`
- `shape_stream_f64_n16_equal-shape-strides_counters`
- `shape_stream_f64_n64_capacity-thrash_counters`
- `shape_stream_f64_n64_equal-shape-strides_counters`

## duplicates

None.

## unexpected

None.

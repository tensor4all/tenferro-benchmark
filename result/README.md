# Benchmark Results

Latest human-facing benchmark reports are kept under this directory: one page per
`target_profile` x device x `suite_id`. Historical reports are available through
git history. The root [README](../README.md) lists the same pages as a generated
table; this file says in which order to read them. The order follows each suite's
own `regime` declaration in `benchmarks/`, so it is not a judgement about the
pages.

CPU pages are pinned-thread reports; GPU pages are per-backend tables. A page
covers one suite only. Do not compare a number from a page in one group with a
number from another group.

## Read these first: throughput and large-matrix pages

Steady-state operation cost of the per-suite measurement pages, including every
suite that declares `regime: throughput`. Read this group when the question is how
fast an operation is.

`nvidia-gpu` (A100):

- [dense](nvidia-gpu/gpu/dense.md) — GPU Dense Benchmarks
- [dense_throughput](nvidia-gpu/gpu/dense_throughput.md) — GPU Dense Throughput Benchmarks (Deferred Batch)
- [einsum](nvidia-gpu/gpu/einsum.md) — GPU Einsum Benchmarks
- [elementwise](nvidia-gpu/gpu/elementwise.md) — GPU Elementwise Chain Benchmarks
- [fft](nvidia-gpu/gpu/fft.md) — CUDA public FFT APIs
- [isolated_ops](nvidia-gpu/gpu/isolated_ops.md) — CUDA public isolated_ops APIs
- [linalg_ad_throughput](nvidia-gpu/gpu/linalg_ad_throughput.md) — GPU Linalg AD Throughput Benchmarks (Deferred Batch)
- [linalg_jvp_vjp](nvidia-gpu/gpu/linalg_jvp_vjp.md) — GPU Linalg JVP/VJP Benchmarks
- [linalg_public_api_throughput](nvidia-gpu/gpu/linalg_public_api_throughput.md) — CUDA public linalg APIs, throughput view (read this one first)
- [lstsq](nvidia-gpu/gpu/lstsq.md) — CUDA public least-squares contract
- [nn](nvidia-gpu/gpu/nn.md) — CUDA public nn APIs
- [permutation](nvidia-gpu/gpu/permutation.md) — GPU Permutation / Materialize Benchmarks
- [reuse_contraction](nvidia-gpu/gpu/reuse_contraction.md) — CUDA public reuse_contraction APIs
- [sparse](nvidia-gpu/gpu/sparse.md) — GPU Sparse Benchmarks
- [structural](nvidia-gpu/gpu/structural.md) — CUDA public structural APIs
- [tensornetwork](nvidia-gpu/gpu/tensornetwork.md) — GPU Tensor Network Benchmarks

`zen5-cpu` (AMD Ryzen AI 9 HX 470):

- [cpu_ops](zen5-cpu/cpu/cpu_ops.md) — CPU Eager and Prepared Operation Comparisons
- [einsum](zen5-cpu/cpu/einsum.md) — CPU Einsum Benchmarks
- [fft](zen5-cpu/cpu/fft.md) — CPU FFT Benchmarks
- [linalg_jvp_vjp](zen5-cpu/cpu/linalg_jvp_vjp.md)
- [permutation](zen5-cpu/cpu/permutation.md) — CPU Permutation / Materialize Benchmarks
- [public_api](zen5-cpu/cpu/public_api.md) — CPU Public API Coverage Benchmarks

## Next: per-call latency and overhead pages

`regime: latency`. Small shapes, where launch and synchronization cost dominates
and the device kernel is a small fraction of the number. These are overhead
diagnostics, not the headline, and they are never comparable with, or averaged
together with, the throughput rows above.

`nvidia-gpu` (A100):

- [linalg_ad_latency](nvidia-gpu/gpu/linalg_ad_latency.md) — GPU Linalg AD Single-Call Latency Benchmarks
- [linalg_public_api_latency](nvidia-gpu/gpu/linalg_public_api_latency.md) — CUDA public linalg APIs, per-call latency and launch overhead
- [perf_issues](nvidia-gpu/gpu/perf_issues.md) — Workloads of open and resolved tenferro-rs performance issues (GPU)

`zen5-cpu` (AMD Ryzen AI 9 HX 470):

- [overhead_smoke](zen5-cpu/cpu/overhead_smoke.md) — CPU overhead smoke: ns/op and allocs/op trend (tenferro-rs #2178)
- [session_matrix](zen5-cpu/cpu/session_matrix.md) — Many small matrices and one-operation batched routes in one CPU session
- [small_work](zen5-cpu/cpu/small_work.md) — CPU small-work public API benchmarks

## Last: diagnostic and investigation reports

Pages with no suite declaration of their own. They answer a specific question
(counters, command wall time, routine workflow and coverage) rather than measure a
suite.

`nvidia-gpu` (A100):

- [command_wall](nvidia-gpu/gpu/command_wall.md)
- [compare_lifetime](nvidia-gpu/gpu/compare_lifetime.md)
- [compare_lifetime_counters](nvidia-gpu/gpu/compare_lifetime_counters.md)
- [linalg_capability](nvidia-gpu/gpu/linalg_capability.md)
- [perf_issues_aa](nvidia-gpu/gpu/perf_issues_aa.md)
- [perf_issues_paired](nvidia-gpu/gpu/perf_issues_paired.md)
- [routine_command_wall](nvidia-gpu/gpu/routine_command_wall.md)
- [routine_controls](nvidia-gpu/gpu/routine_controls.md)
- [routine_coverage](nvidia-gpu/gpu/routine_coverage.md)
- [routine_workflow](nvidia-gpu/gpu/routine_workflow.md)
- [shape_stream](nvidia-gpu/gpu/shape_stream.md)

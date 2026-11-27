# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261127_191419`.

### run_t1

```yaml
timestamp: '2026-11-27T19:14:19Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 7da876a214acb65eb9ea099adef67b41ef8efc98
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=amd-cpu /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 5
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

### run_t4

```yaml
timestamp: '2026-11-27T19:14:52Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 7da876a214acb65eb9ea099adef67b41ef8efc98
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=amd-cpu /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 5
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

## Case status

- Manifest: `cpu/session_matrix` version 5, coverage `quick`, effort `standard`
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 4.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 4 |

- Noisy: `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`, `stream_f64_fresh_len32_einsum_alloc[faer]@t4`, `stream_f64_strides_len32_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.77 | 0.06 | 2705.08 | 2.0% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 2.36 | 0.08 | 2304.69 | 3.9% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 3.29 | 0.13 | 3212.89 | 2.7% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 2.44 | 0.03 | 2382.81 | 5.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 2.36 | 0.02 | 2304.69 | 1.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.35 | 0.06 | 1318.36 | 7.3% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.81 | 0.01 | 2744.14 | 2.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.35 | 0.04 | 1318.36 | 7.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 5.35 | 0.09 | 5224.61 | 1.2% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 4.16 | 0.04 | 4062.50 | 2.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 9.05 | 0.18 | 8837.89 | 1.3% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 4.16 | 0.04 | 4062.50 | 2.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 2.42 | 0.02 | 2363.28 | 0.9% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.46 | 0.11 | 1425.78 | 7.8% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.81 | 0.01 | 2744.14 | 1.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.49 | 0.05 | 1455.08 | 7.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.46 | 0.02 | 2402.34 | 0.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 1.53 | 0.03 | 1494.14 | 6.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.85 | 0.01 | 2783.20 | 0.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 1.59 | 0.03 | 1552.73 | 7.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 4.30 | 0.12 | 4199.22 | 2.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 17.11 | 0.43 | 16708.98 | 1.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 4.79 | 0.09 | 4677.73 | 3.4% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 16.58 | 0.13 | 16191.41 | 1.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.39 | 0.03 | 1357.42 | 1.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 12.48 | 0.21 | 12187.50 | 1.2% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.88 | 0.02 | 1835.94 | 0.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 12.53 | 0.18 | 12236.33 | 3.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 13.23 | 0.66 | 12919.92 | 7.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 25.18 | 0.28 | 24589.84 | 1.0% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 17.13 | 0.18 | 16728.52 | 1.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 25.27 | 0.07 | 24677.73 | 0.7% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.48 | 0.02 | 1445.31 | 2.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 12.63 | 0.06 | 12333.98 | 2.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.96 | 0.06 | 1914.06 | 5.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 12.64 | 0.15 | 12343.75 | 1.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.04 | 0.04 | 1992.19 | 1.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 17.07 | 0.10 | 16669.92 | 2.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.53 | 0.07 | 2470.70 | 1.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 17.28 | 0.08 | 16875.00 | 0.6% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 281863.33 | 2404.50 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 7 | 123135.00 | 1346.64 | 1.5% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 7 | 126772.86 | 1104.07 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 54752.77 | 1113.15 | 13.8% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 267277.00 | 2680.33 | 1.1% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 8 | 95707.38 | 1410.25 | 2.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 49195.20 | 925.33 | 1.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 21 | 24300.24 | 453.14 | 1.1% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 122842.25 | 2771.38 | 2.2% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 4 | 181763.50 | 4632.62 | 2.5% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 118965.25 | 4969.12 | 2.7% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 171921.00 | 7121.83 | 3.7% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 28 | 18299.14 | 554.09 | 2.5% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 31 | 19083.94 | 2145.98 | 8.9% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 137 | 4098.42 | 103.85 | 3.3% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 111 | 5266.36 | 654.81 | 6.9% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 8 | 78305.12 | 4191.94 | 3.9% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 17 | 31642.76 | 829.47 | 1.6% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 9 | 69767.11 | 2161.28 | 3.3% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 23 | 27286.65 | 645.91 | 3.9% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 266431.00 | 4818.00 | 1.2% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 8 | 98761.88 | 1278.38 | 1.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 50428.47 | 1167.77 | 1.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 23 | 25170.22 | 898.91 | 2.3% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 26 | 24487.92 | 937.73 | 6.7% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 20 | 29858.70 | 5678.35 | 10.6% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 165 | 3153.42 | 256.88 | 6.4% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 139 | 3455.77 | 538.82 | 9.0% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 57 | 10505.25 | 420.93 | 2.5% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 42 | 12652.57 | 1935.64 | 8.9% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.13 | 0.00 | 0.140 | 0.151 | 4.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.13 | 0.00 | 0.149 | 0.173 | 8.3% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.27 | 0.02 | 0.296 | 0.336 | 8.2% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.23 | 0.00 | 0.237 | 0.242 | 2.2% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.54 | 0.02 | 1.554 | 1.562 | 0.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 1.45 | 0.04 | 1.548 | 1.559 | 2.8% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 4.90 | 0.21 | 5.004 | 5.155 | 2.6% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.90 | 0.07 | 1.971 | 2.146 | 4.4% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.18 | 0.00 | 0.196 | 0.203 | 9.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.18 | 0.01 | 0.195 | 0.197 | 7.8% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.30 | 0.07 | 0.352 | 0.358 | 11.3% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.26 | 0.05 | 0.305 | 0.332 | 10.9% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.53 | 0.01 | 1.557 | 1.577 | 1.1% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.56 | 0.02 | 1.602 | 1.894 | 5.6% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 4.86 | 0.10 | 5.176 | 5.416 | 3.6% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.93 | 0.04 | 2.030 | 2.057 | 2.5% | passed |

### Separate-process cache observations

Public EagerRuntime extension-store counters only. Provider GEMM-analysis/indexed-plan caches and runtime prepared caches are outside this scope. No public plan-build counter exists; lookup misses are not treated as builds. These processes collect no latency samples. Hits/misses/evictions are measured-pass deltas; retained bytes are a logical estimate, not process RSS.

| Case | Threads | Hits | Misses | Evictions | Final entries | Final retained bytes | Builds | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| eager_stream_f64_fixed_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 4 | 5792 | unavailable | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | 1 | 0 | 1920 | 1920 | 256 | 370688 | unavailable | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 4 | 5792 | unavailable | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | 4 | 0 | 1920 | 1920 | 256 | 370688 | unavailable | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |

Wall observations `wall_collection_t1.json`:
```json
{
  "latency_case_processes_seconds": 29.793825519000166,
  "separate_cache_diagnostics_seconds": 0.38974345099995844,
  "collection_total_seconds": 30.183569290000378
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 30.334861280999576,
  "separate_cache_diagnostics_seconds": 0.4129626140002074,
  "collection_total_seconds": 30.74782409999989
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 70.688529363,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 3.119095736,
  "validation_metadata_and_collection_seconds": 67.508927309,
  "command_before_report_seconds": 70.628023045
}
```

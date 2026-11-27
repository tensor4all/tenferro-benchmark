# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261127_184247`.

### run_t1

```yaml
timestamp: '2026-11-27T18:42:47Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 7da876a214acb65eb9ea099adef67b41ef8efc98
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=mac-cpu /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_cpu_session.sh
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
timestamp: '2026-11-27T18:43:19Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 7da876a214acb65eb9ea099adef67b41ef8efc98
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=mac-cpu /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_cpu_session.sh
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 38.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 38 |

- Noisy: `matmul_n2_count1024[pytorch]@t1`, `matmul_n4_count1024[pytorch]@t1`, `matmul_n8_count1024[pytorch]@t1`, `matmul_n16_count1024[pytorch]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[pytorch]@t1`, `solve_n4_count1024[pytorch]@t1`, `solve_n8_count1024[pytorch]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n16_count1024[pytorch]@t1`, `solve_n32_count1024[faer]@t1`, `solve_n32_count1024[pytorch]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n2_count1024[pytorch]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n4_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4`, `solve_n8_count1024[faer]@t4`, `solve_n8_count1024[pytorch]@t4`, `solve_n16_count1024[faer]@t4`, `solve_n16_count1024[pytorch]@t4`, `solve_n32_count1024[faer]@t4`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t4`, `stream_f64_fresh_len32_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 1.41 | 0.04 | 1376.95 | 3.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.31 | 0.60 | 1279.30 | 56.8% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 1.41 | 0.10 | 1376.95 | 4.4% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.21 | 0.08 | 1181.64 | 29.3% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.92 | 0.04 | 898.44 | 8.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.71 | 0.07 | 693.36 | 25.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.18 | 0.18 | 1152.34 | 35.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.70 | 0.09 | 683.59 | 19.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 4.10 | 0.31 | 4003.91 | 35.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 1.64 | 0.12 | 1601.56 | 6.1% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 4.05 | 0.26 | 3955.08 | 4.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.65 | 0.06 | 1611.33 | 3.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 0.80 | 0.01 | 781.25 | 2.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.83 | 0.14 | 810.55 | 62.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 0.96 | 0.13 | 937.50 | 11.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.74 | 0.12 | 722.66 | 16.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 0.93 | 0.04 | 908.20 | 3.8% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.86 | 0.10 | 839.84 | 11.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 0.98 | 0.17 | 957.03 | 9.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.93 | 0.06 | 908.20 | 4.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 3.57 | 1.36 | 3486.33 | 23.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 10.70 | 0.74 | 10449.22 | 13.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 3.23 | 0.87 | 3154.30 | 19.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 11.32 | 2.06 | 11054.69 | 11.0% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.90 | 0.04 | 878.91 | 5.0% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 9.67 | 0.70 | 9443.36 | 28.9% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.19 | 0.31 | 1162.11 | 23.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 8.66 | 0.62 | 8457.03 | 8.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 11.93 | 6.37 | 11650.39 | 34.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 17.22 | 2.92 | 16816.41 | 23.2% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 10.41 | 1.35 | 10166.02 | 18.2% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 21.10 | 2.26 | 20605.47 | 7.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.02 | 0.10 | 996.09 | 8.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 9.21 | 2.09 | 8994.14 | 14.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.26 | 0.42 | 1230.47 | 17.8% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 8.84 | 0.74 | 8632.81 | 6.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.29 | 0.11 | 1259.77 | 5.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 10.28 | 2.51 | 10039.06 | 18.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 1.92 | 0.53 | 1875.00 | 41.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 10.50 | 2.49 | 10253.91 | 17.1% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 62622.20 | 2513.90 | 3.5% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 75541.69 | 4120.19 | 4.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 58776.79 | 2428.54 | 4.8% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 68272.38 | 4141.04 | 3.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 20 | 34812.50 | 1352.05 | 9.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 17 | 40558.82 | 9729.12 | 12.1% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 18 | 26831.06 | 150.44 | 2.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 23 | 28592.39 | 1030.78 | 3.0% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 222486.00 | 4382.00 | 2.2% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 1 | 411166.00 | 153312.50 | 21.8% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 216198.00 | 2760.25 | 2.0% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 1 | 406125.00 | 119354.50 | 16.9% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 163 | 3313.39 | 1043.07 | 49.2% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 69 | 3345.41 | 546.79 | 35.0% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 43 | 2470.93 | 73.16 | 17.4% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 72 | 2556.72 | 157.40 | 5.4% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 18 | 48196.78 | 2047.47 | 3.3% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 9 | 61810.22 | 17828.67 | 23.1% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 18 | 46916.67 | 794.03 | 2.1% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 13 | 50128.23 | 10584.92 | 18.2% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 18 | 35148.17 | 682.89 | 1.8% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 21 | 39394.86 | 2830.36 | 7.2% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 27 | 26191.33 | 996.93 | 4.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 19 | 27927.63 | 783.97 | 5.9% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 16 | 17216.12 | 342.47 | 2.4% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 12 | 18312.50 | 1156.33 | 4.3% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 77 | 1497.83 | 5.14 | 2.5% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 101 | 1610.15 | 93.03 | 4.5% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 18 | 8682.83 | 791.69 | 7.1% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 45 | 8526.84 | 411.12 | 9.3% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.08 | 0.02 | 0.125 | 0.149 | 29.1% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.18 | 0.11 | 0.265 | 0.607 | 67.2% | passed (NOISY; inconclusive) |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.18 | 0.03 | 0.220 | 0.564 | 49.8% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.11 | 0.00 | 0.118 | 0.125 | 2.5% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 0.76 | 0.04 | 1.096 | 1.742 | 31.6% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 0.69 | 0.03 | 0.715 | 0.756 | 3.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 4.00 | 1.16 | 5.055 | 5.596 | 16.7% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 0.92 | 0.03 | 0.993 | 1.043 | 4.4% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.08 | 0.01 | 0.083 | 0.084 | 5.5% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.07 | 0.00 | 0.073 | 0.078 | 2.1% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.20 | 0.04 | 0.379 | 0.405 | 33.4% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.12 | 0.00 | 0.129 | 0.131 | 4.0% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 0.77 | 0.02 | 0.781 | 0.957 | 6.5% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 0.81 | 0.07 | 0.846 | 0.864 | 4.9% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 3.07 | 0.13 | 3.186 | 3.469 | 4.0% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.00 | 0.03 | 1.020 | 1.030 | 1.9% | passed |

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
  "latency_case_processes_seconds": 26.939290833000086,
  "separate_cache_diagnostics_seconds": 0.27869933399961155,
  "collection_total_seconds": 27.217990541999825
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 26.885148374999517,
  "separate_cache_diagnostics_seconds": 0.2854723330001434,
  "collection_total_seconds": 27.170621125000253
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 69.456868834,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 5.047104917,
  "validation_metadata_and_collection_seconds": 64.301704292,
  "command_before_report_seconds": 69.348809209
}
```

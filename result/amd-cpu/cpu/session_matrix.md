# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261125_192145`.

### run_t1

```yaml
timestamp: '2026-11-25T19:21:46Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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
timestamp: '2026-11-25T19:22:25Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 15.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 15 |

- Noisy: `matmul_n16_count1024[faer]@t4`, `solve_n32_count1024[pytorch]@t4`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4`, `hadamard_f64_m64n64_dot_alloc[faer]@t4`, `hadamard_f64_m64n64_einsum_alloc[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`, `stream_f64_mixed_len32_einsum_alloc[faer]@t4`, `stream_f64_fresh_len32_einsum_alloc[faer]@t4`, `stream_f64_strides_len32_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.00 | 0.02 | 1953.12 | 1.3% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.35 | 0.03 | 3271.48 | 3.6% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 2.63 | 0.05 | 2568.36 | 16.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.46 | 0.14 | 3378.91 | 4.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.61 | 0.01 | 1572.27 | 0.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.90 | 0.07 | 1855.47 | 5.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.17 | 0.01 | 2119.14 | 2.2% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.95 | 0.09 | 1904.30 | 6.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 4.18 | 0.11 | 4082.03 | 2.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 7.57 | 0.13 | 7392.58 | 9.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 4.56 | 0.21 | 4453.12 | 2.9% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 10.53 | 0.21 | 10283.20 | 1.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.63 | 0.02 | 1591.80 | 1.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.83 | 0.03 | 1787.11 | 6.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.23 | 0.01 | 2177.73 | 0.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.94 | 0.07 | 1894.53 | 5.7% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.67 | 0.01 | 1630.86 | 0.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.12 | 0.10 | 2070.31 | 7.3% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.21 | 0.01 | 2158.20 | 0.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.31 | 0.10 | 2255.86 | 5.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 4.49 | 0.10 | 4384.77 | 1.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 21.02 | 0.38 | 20527.34 | 1.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 5.17 | 0.13 | 5048.83 | 1.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 21.56 | 0.48 | 21054.69 | 2.2% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.62 | 0.03 | 1582.03 | 1.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 16.48 | 0.26 | 16093.75 | 1.4% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.28 | 0.03 | 2226.56 | 1.1% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 16.11 | 0.33 | 15732.42 | 1.4% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 13.62 | 0.25 | 13300.78 | 1.0% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 36.97 | 0.38 | 36103.52 | 5.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 14.43 | 0.13 | 14091.80 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 52.51 | 0.85 | 51279.30 | 11.8% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.75 | 0.03 | 1708.98 | 1.0% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 16.76 | 0.25 | 16367.19 | 1.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.45 | 0.01 | 2392.58 | 2.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 16.95 | 0.22 | 16552.73 | 0.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.39 | 0.01 | 2333.98 | 0.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 17.67 | 0.16 | 17255.86 | 0.8% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 3.04 | 0.01 | 2968.75 | 2.3% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 17.83 | 0.35 | 17412.11 | 2.0% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 294364.67 | 2730.00 | 0.8% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 2 | 299903.50 | 10679.75 | 10.2% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 166267.60 | 4097.50 | 1.9% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 4 | 170582.50 | 42757.12 | 11.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 221948.25 | 1506.50 | 1.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 3 | 234663.67 | 4443.17 | 9.0% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 16 | 54812.69 | 609.88 | 1.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 11 | 56270.27 | 32244.59 | 23.8% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 209106.75 | 1189.75 | 1.4% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 239158.67 | 4106.00 | 1.9% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 210251.67 | 2236.17 | 0.9% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 239155.33 | 2760.17 | 1.0% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 49 | 15338.78 | 253.03 | 1.3% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 34 | 16311.53 | 5968.59 | 16.3% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 139 | 4684.77 | 133.95 | 1.4% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 88 | 8075.49 | 2233.07 | 18.1% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 7 | 78809.57 | 2803.21 | 2.1% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 10 | 68626.80 | 4196.80 | 4.5% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 17 | 53769.00 | 1829.91 | 2.2% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 13 | 46276.92 | 4825.54 | 9.6% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 227546.25 | 3296.12 | 1.1% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 2 | 227714.00 | 16100.00 | 14.9% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 54166.87 | 686.57 | 1.3% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 9 | 59048.89 | 34872.67 | 24.8% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 20 | 28355.25 | 1851.95 | 3.5% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 17 | 42238.06 | 12681.26 | 17.6% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 188 | 2757.79 | 142.10 | 3.0% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 115 | 4360.90 | 225.77 | 11.7% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 46 | 11263.37 | 397.37 | 2.8% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 39 | 17831.41 | 5462.10 | 17.6% | passed (NOISY) |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.10 | 0.00 | 0.113 | 0.128 | 9.3% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.10 | 0.00 | 0.102 | 0.131 | 7.5% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.25 | 0.01 | 0.272 | 0.283 | 6.4% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.19 | 0.00 | 0.217 | 0.224 | 5.9% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.16 | 0.01 | 1.172 | 1.180 | 0.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 1.18 | 0.01 | 1.185 | 1.219 | 1.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 5.80 | 0.20 | 6.142 | 6.240 | 3.1% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.58 | 0.03 | 1.611 | 1.641 | 1.5% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.19 | 0.00 | 0.198 | 0.199 | 3.1% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.19 | 0.00 | 0.203 | 0.290 | 13.8% | passed (NOISY; inconclusive) |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.36 | 0.13 | 0.410 | 0.420 | 19.7% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.31 | 0.08 | 0.330 | 0.350 | 17.3% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.37 | 0.02 | 1.388 | 1.421 | 1.2% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.40 | 0.01 | 1.417 | 1.430 | 0.8% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 6.23 | 0.30 | 6.513 | 6.648 | 3.2% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.78 | 0.02 | 1.796 | 1.837 | 1.0% | passed |

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
  "latency_case_processes_seconds": 36.134639749999224,
  "separate_cache_diagnostics_seconds": 0.40765450000071723,
  "collection_total_seconds": 36.542294449999645
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 38.22117046099993,
  "separate_cache_diagnostics_seconds": 0.4547386119993462,
  "collection_total_seconds": 38.67590931399991
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 85.313593429,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 3.085881384,
  "validation_metadata_and_collection_seconds": 82.163746124,
  "command_before_report_seconds": 85.249627508
}
```

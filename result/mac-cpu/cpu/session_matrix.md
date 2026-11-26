# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261126_184711`.

### run_t1

```yaml
timestamp: '2026-11-26T18:47:11Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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
timestamp: '2026-11-26T18:47:42Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 52.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 52 |

- Noisy: `matmul_n2_count1024[pytorch]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n16_count1024[pytorch]@t1`, `matmul_n32_count1024[pytorch]@t1`, `solve_n4_count1024[pytorch]@t1`, `solve_n8_count1024[pytorch]@t1`, `solve_n16_count1024[pytorch]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n2_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `solve_n4_count1024[faer]@t4`, `solve_n4_count1024[pytorch]@t4`, `solve_n8_count1024[faer]@t4`, `solve_n16_count1024[pytorch]@t4`, `solve_n32_count1024[faer]@t4`, `solve_n32_count1024[pytorch]@t4`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t4` (+12 more)

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 1.44 | 0.04 | 1406.25 | 24.4% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.18 | 0.07 | 1152.34 | 10.6% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 1.32 | 0.04 | 1289.06 | 4.8% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.54 | 0.14 | 1503.91 | 18.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.83 | 0.06 | 810.55 | 5.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.75 | 0.06 | 732.42 | 26.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.06 | 0.08 | 1035.16 | 26.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.84 | 0.06 | 820.31 | 11.9% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 4.05 | 0.26 | 3955.08 | 3.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 1.82 | 0.10 | 1777.34 | 24.7% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 4.02 | 0.18 | 3925.78 | 8.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.82 | 0.14 | 1777.34 | 8.2% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 0.90 | 0.03 | 878.91 | 3.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.79 | 0.04 | 771.48 | 3.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 0.97 | 0.06 | 947.27 | 3.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.76 | 0.05 | 742.19 | 4.0% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 0.94 | 0.06 | 917.97 | 13.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.76 | 0.08 | 742.19 | 8.8% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 0.93 | 0.02 | 908.20 | 3.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.81 | 0.08 | 791.02 | 9.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.72 | 0.04 | 2656.25 | 6.4% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 10.39 | 1.36 | 10146.48 | 15.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 3.05 | 0.23 | 2978.52 | 5.7% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 12.38 | 2.61 | 12089.84 | 12.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.92 | 0.05 | 898.44 | 4.0% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 8.42 | 0.35 | 8222.66 | 4.4% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.15 | 0.12 | 1123.05 | 7.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 9.72 | 0.67 | 9492.19 | 4.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 11.18 | 0.54 | 10917.97 | 23.3% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 19.07 | 1.42 | 18623.05 | 8.7% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 11.53 | 1.75 | 11259.77 | 14.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 20.15 | 1.67 | 19677.73 | 14.7% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 0.93 | 0.04 | 908.20 | 3.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 8.42 | 2.61 | 8222.66 | 16.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.20 | 0.10 | 1171.88 | 14.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 9.83 | 1.21 | 9599.61 | 22.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.31 | 0.06 | 1279.30 | 3.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 9.34 | 0.76 | 9121.09 | 10.1% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 1.57 | 0.23 | 1533.20 | 20.8% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 10.72 | 0.77 | 10468.75 | 4.8% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 11 | 81481.00 | 31600.32 | 44.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 10 | 105670.90 | 91858.30 | 80.7% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 78208.25 | 25302.12 | 17.2% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 6 | 185507.00 | 93472.17 | 47.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 10 | 38345.80 | 6795.75 | 10.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 3 | 44000.00 | 24826.33 | 41.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 22 | 28789.77 | 2162.86 | 7.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 24 | 30689.25 | 3299.46 | 8.6% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 2 | 243042.00 | 10229.00 | 4.3% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 2 | 334187.50 | 140812.75 | 27.0% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 238055.33 | 16972.00 | 10.7% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 1 | 564334.00 | 706479.00 | 79.8% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 151 | 3091.06 | 285.32 | 4.7% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 138 | 3290.76 | 446.26 | 12.8% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 98 | 2585.03 | 222.79 | 6.3% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 89 | 2950.84 | 672.52 | 18.3% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 16 | 53427.12 | 2516.91 | 5.4% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 7 | 46922.57 | 23758.86 | 24.1% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 14 | 55026.79 | 5918.18 | 7.3% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 7 | 42482.14 | 20241.07 | 25.7% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 16 | 41114.56 | 4947.91 | 9.2% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 43317.69 | 5540.38 | 12.7% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 16 | 29130.25 | 1334.59 | 6.3% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 27 | 32037.04 | 3309.41 | 5.1% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 10 | 38825.00 | 40783.30 | 69.6% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 12 | 19270.83 | 1244.83 | 10.8% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 23 | 1815.22 | 207.43 | 58.7% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 126 | 1729.83 | 187.66 | 6.4% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 6 | 9569.50 | 3500.00 | 55.1% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 38 | 14367.32 | 10946.82 | 38.4% | passed (NOISY) |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.07 | 0.00 | 0.089 | 0.103 | 12.9% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.08 | 0.01 | 0.139 | 0.218 | 39.9% | passed (NOISY; inconclusive) |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.38 | 0.34 | 0.643 | 0.899 | 55.7% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.22 | 0.26 | 0.409 | 0.418 | 50.9% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.30 | 1.10 | 2.505 | 2.616 | 41.4% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 0.88 | 0.83 | 2.135 | 2.624 | 50.7% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 4.28 | 2.21 | 7.597 | 10.302 | 40.0% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.14 | 0.28 | 2.574 | 3.214 | 47.8% | passed (NOISY; inconclusive) |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.08 | 0.01 | 0.111 | 0.125 | 19.1% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.08 | 0.00 | 0.090 | 0.173 | 29.4% | passed (NOISY; inconclusive) |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.20 | 0.02 | 0.226 | 0.285 | 14.8% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.13 | 0.01 | 0.157 | 0.209 | 16.3% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 0.85 | 0.08 | 1.035 | 2.080 | 33.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 0.89 | 0.19 | 1.729 | 2.094 | 39.1% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 4.64 | 3.80 | 9.319 | 9.466 | 41.9% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.05 | 0.07 | 1.179 | 1.198 | 6.3% | passed |

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
  "latency_case_processes_seconds": 26.44324470799984,
  "separate_cache_diagnostics_seconds": 0.44598833399959403,
  "collection_total_seconds": 26.889233375000003
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 26.527972917000625,
  "separate_cache_diagnostics_seconds": 0.3858912079995207,
  "collection_total_seconds": 26.913865041999998
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 66.693236,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 4.376605375,
  "validation_metadata_and_collection_seconds": 62.177260208,
  "command_before_report_seconds": 66.553865583
}
```

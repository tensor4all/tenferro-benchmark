# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261125_185034`.

### run_t1

```yaml
timestamp: '2026-11-25T18:50:35Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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
timestamp: '2026-11-25T18:51:05Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 74775e4a2ad3480fb9ee9f859e0037b8cac731f2
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 24.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 24 |

- Noisy: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[pytorch]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n8_count1024[pytorch]@t1`, `solve_n16_count1024[pytorch]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n4_count1024[pytorch]@t4`, `matmul_n32_count1024[pytorch]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[pytorch]@t4`, `solve_n8_count1024[pytorch]@t4`, `solve_n16_count1024[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4`, `hadamard_f64_m64n64_dot_alloc[faer]@t4`, `stream_f64_fixed_len32_einsum_alloc[faer]@t4`, `stream_f64_strides_len32_einsum_alloc[faer]@t4`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t4`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t4`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 1.40 | 0.05 | 1367.19 | 4.0% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.23 | 0.03 | 1201.17 | 2.1% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 1.42 | 0.07 | 1386.72 | 3.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.11 | 0.10 | 1083.98 | 6.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.83 | 0.12 | 810.55 | 10.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.65 | 0.02 | 634.77 | 4.2% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.02 | 0.06 | 996.09 | 15.9% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.65 | 0.02 | 634.77 | 2.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 3.86 | 0.33 | 3769.53 | 5.9% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 1.88 | 0.08 | 1835.94 | 8.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 3.59 | 0.13 | 3505.86 | 2.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.84 | 0.12 | 1796.88 | 13.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 0.84 | 0.01 | 820.31 | 1.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.70 | 0.07 | 683.59 | 11.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.04 | 0.32 | 1015.62 | 24.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.70 | 0.05 | 683.59 | 32.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 0.92 | 0.05 | 898.44 | 4.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.92 | 0.04 | 898.44 | 3.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 0.96 | 0.02 | 937.50 | 1.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.87 | 0.03 | 849.61 | 2.7% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.68 | 0.06 | 2617.19 | 2.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 10.13 | 1.43 | 9892.58 | 12.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 2.99 | 0.29 | 2919.92 | 64.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 9.78 | 0.22 | 9550.78 | 2.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.92 | 0.01 | 898.44 | 1.1% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 8.93 | 0.16 | 8720.70 | 1.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.09 | 0.21 | 1064.45 | 23.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 7.76 | 0.10 | 7578.12 | 0.7% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 10.49 | 1.07 | 10244.14 | 5.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 18.08 | 0.82 | 17656.25 | 3.7% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 9.67 | 0.32 | 9443.36 | 8.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 17.60 | 0.25 | 17187.50 | 2.7% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.01 | 0.03 | 986.33 | 2.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 8.67 | 0.09 | 8466.80 | 1.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.18 | 0.06 | 1152.34 | 7.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 9.05 | 2.79 | 8837.89 | 17.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.49 | 0.18 | 1455.08 | 11.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 10.10 | 2.36 | 9863.28 | 15.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 1.41 | 0.02 | 1376.95 | 1.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 8.87 | 2.52 | 8662.11 | 18.0% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 66958.36 | 717.29 | 5.9% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 70948.69 | 2519.23 | 2.5% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 70255.93 | 7336.29 | 6.6% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 12 | 67357.67 | 2588.54 | 3.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 21 | 34918.67 | 306.55 | 1.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 21 | 35861.10 | 1547.60 | 3.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 28 | 27049.11 | 422.61 | 0.9% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 27 | 27729.96 | 1110.33 | 2.6% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 231427.25 | 3046.88 | 1.2% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 2 | 253208.00 | 36739.50 | 11.7% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 225250.00 | 1923.67 | 1.6% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 186708.33 | 38771.00 | 15.7% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 127 | 2965.22 | 16.41 | 1.9% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 120 | 3214.58 | 159.72 | 3.3% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 87 | 2456.90 | 4.07 | 1.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 50 | 2569.16 | 149.17 | 3.9% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 15 | 48136.13 | 1393.00 | 3.3% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 12 | 29604.17 | 3781.25 | 9.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 17 | 48058.82 | 634.82 | 2.1% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 16 | 28593.75 | 3486.97 | 8.1% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 22 | 36859.86 | 458.34 | 1.5% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 38895.81 | 1748.75 | 3.1% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 26 | 27187.50 | 182.67 | 1.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 25 | 28818.36 | 821.66 | 3.0% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 14 | 19232.14 | 1014.86 | 33.0% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 16 | 19289.06 | 1570.28 | 6.8% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 92 | 1610.96 | 37.59 | 3.9% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 139 | 1840.53 | 328.38 | 12.8% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 22 | 9058.68 | 382.55 | 4.3% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 19 | 8576.74 | 256.55 | 6.1% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.07 | 0.01 | 0.083 | 0.094 | 9.5% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.07 | 0.00 | 0.072 | 0.081 | 3.3% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.18 | 0.03 | 0.218 | 0.314 | 20.2% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.12 | 0.00 | 0.123 | 0.127 | 3.2% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 0.73 | 0.01 | 0.759 | 0.781 | 2.3% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 0.73 | 0.02 | 0.754 | 0.759 | 1.7% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 3.04 | 0.34 | 3.954 | 4.359 | 14.2% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 0.96 | 0.04 | 1.034 | 1.193 | 6.3% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.09 | 0.01 | 0.112 | 0.119 | 12.9% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.08 | 0.01 | 0.082 | 0.089 | 6.2% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.19 | 0.01 | 0.197 | 0.222 | 8.9% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.14 | 0.03 | 0.156 | 0.168 | 11.7% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 0.76 | 0.03 | 0.790 | 0.791 | 2.3% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 0.90 | 0.34 | 1.264 | 1.311 | 20.1% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 3.43 | 0.74 | 4.591 | 5.120 | 16.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.04 | 0.34 | 1.429 | 1.716 | 18.9% | passed (NOISY; inconclusive) |

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
  "latency_case_processes_seconds": 24.895515125000202,
  "separate_cache_diagnostics_seconds": 0.2691434999997,
  "collection_total_seconds": 25.164658916000008
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 24.3385916670004,
  "separate_cache_diagnostics_seconds": 0.35075920799954474,
  "collection_total_seconds": 24.689351167000495
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 64.307087417,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 3.9245305,
  "validation_metadata_and_collection_seconds": 60.263665334,
  "command_before_report_seconds": 64.188195834
}
```

# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261124_184309`.

### run_t1

```yaml
timestamp: '2026-11-24T18:43:10Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 9185e8ca26c3fc600458361d0f477495a717daf3
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
timestamp: '2026-11-24T18:43:43Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 9185e8ca26c3fc600458361d0f477495a717daf3
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 61.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 61 |

- Noisy: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n4_count1024[pytorch]@t1`, `matmul_n8_count1024[pytorch]@t1`, `matmul_n32_count1024[faer]@t1`, `matmul_n32_count1024[pytorch]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n2_count1024[pytorch]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n4_count1024[pytorch]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n8_count1024[pytorch]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `solve_n32_count1024[pytorch]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n2_count1024[pytorch]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n4_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `solve_n2_count1024[pytorch]@t4`, `solve_n4_count1024[pytorch]@t4`, `solve_n8_count1024[pytorch]@t4`, `solve_n16_count1024[faer]@t4`, `solve_n16_count1024[pytorch]@t4`, `solve_n32_count1024[faer]@t4`, `solve_n32_count1024[pytorch]@t4`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4` (+21 more)

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 1.36 | 0.15 | 1328.12 | 7.6% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.16 | 0.15 | 1132.81 | 8.0% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 1.91 | 0.03 | 1865.23 | 1.3% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.14 | 0.13 | 1113.28 | 29.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 0.90 | 0.29 | 878.91 | 39.9% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.84 | 0.07 | 820.31 | 8.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.57 | 0.21 | 1533.20 | 17.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.64 | 0.16 | 625.00 | 14.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 4.60 | 1.96 | 4492.19 | 22.3% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 2.13 | 0.29 | 2080.08 | 34.3% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 4.35 | 0.13 | 4248.05 | 3.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.50 | 0.06 | 1464.84 | 6.2% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 0.83 | 0.07 | 810.55 | 11.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.72 | 0.09 | 703.12 | 31.9% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.23 | 1.51 | 2177.73 | 51.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.71 | 0.04 | 693.36 | 16.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.01 | 0.02 | 986.33 | 3.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.99 | 0.06 | 966.80 | 18.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 1.45 | 0.05 | 1416.02 | 6.3% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.75 | 0.06 | 732.42 | 5.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 4.97 | 1.55 | 4853.52 | 32.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 11.38 | 1.15 | 11113.28 | 7.2% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 3.38 | 0.54 | 3300.78 | 37.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 11.32 | 0.92 | 11054.69 | 10.1% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.01 | 0.30 | 986.33 | 40.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 8.55 | 0.46 | 8349.61 | 20.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 1.58 | 0.06 | 1542.97 | 3.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 9.25 | 0.77 | 9033.20 | 16.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 12.13 | 3.86 | 11845.70 | 23.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 20.86 | 3.70 | 20371.09 | 14.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 12.63 | 2.86 | 12333.98 | 29.2% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 23.44 | 1.73 | 22890.62 | 13.0% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.16 | 0.11 | 1132.81 | 25.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 8.77 | 4.32 | 8564.45 | 28.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 1.48 | 0.04 | 1445.31 | 2.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 9.22 | 0.99 | 9003.91 | 20.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.46 | 0.30 | 1425.78 | 24.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 10.05 | 1.43 | 9814.45 | 12.8% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 1.98 | 0.09 | 1933.59 | 6.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 9.04 | 1.04 | 8828.12 | 15.4% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 10 | 89408.30 | 93998.00 | 42.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 4 | 148208.50 | 102484.25 | 37.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 13 | 62355.77 | 2629.85 | 5.3% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 83057.69 | 24823.73 | 20.2% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 41466.67 | 3941.63 | 38.7% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 18 | 45807.89 | 37468.72 | 43.1% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 33 | 28477.27 | 5319.45 | 20.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 30 | 35420.83 | 12206.25 | 38.2% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 242125.00 | 59784.83 | 25.7% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 1 | 355917.00 | 211499.50 | 217.7% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 224458.33 | 24069.67 | 6.8% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 1 | 458875.00 | 106645.50 | 27.2% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 82 | 2967.99 | 118.90 | 2.6% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 52 | 3978.37 | 299.29 | 22.5% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 247 | 2494.43 | 30.79 | 28.9% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 56 | 4000.73 | 1550.23 | 34.7% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 13 | 48067.31 | 1299.69 | 2.4% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 7 | 53815.43 | 17604.29 | 35.5% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 19 | 47730.26 | 2234.61 | 4.3% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 10 | 52383.30 | 18983.35 | 44.0% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 17 | 37899.47 | 965.68 | 3.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 15 | 68419.47 | 51604.20 | 37.7% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 32 | 26562.50 | 680.36 | 2.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 19 | 34333.37 | 4644.76 | 17.0% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 12 | 19489.58 | 531.25 | 2.1% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 5 | 24275.00 | 8312.60 | 30.9% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 43 | 1585.28 | 42.15 | 8.3% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 50 | 7441.66 | 1683.75 | 30.4% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 13 | 8326.92 | 350.92 | 6.2% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 30 | 15806.97 | 16184.72 | 46.9% | passed (NOISY) |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.07 | 0.02 | 0.106 | 0.162 | 30.6% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.07 | 0.00 | 0.073 | 0.074 | 2.1% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.77 | 0.60 | 1.581 | 2.335 | 63.0% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.16 | 0.21 | 0.528 | 0.610 | 61.1% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 0.77 | 0.06 | 1.010 | 1.306 | 19.2% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 0.75 | 0.24 | 1.218 | 2.078 | 39.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 3.29 | 0.66 | 4.038 | 4.998 | 16.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.11 | 0.04 | 1.489 | 1.763 | 18.1% | passed (NOISY; inconclusive) |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.31 | 0.23 | 0.412 | 0.425 | 50.7% | passed (NOISY; inconclusive) |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.31 | 0.01 | 0.328 | 0.339 | 3.8% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.31 | 0.14 | 0.704 | 1.209 | 60.0% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.21 | 0.07 | 0.276 | 0.284 | 17.9% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.14 | 0.51 | 2.126 | 2.690 | 35.7% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.46 | 1.30 | 2.691 | 2.913 | 39.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 4.57 | 1.37 | 6.484 | 8.458 | 28.3% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.83 | 0.66 | 3.656 | 4.088 | 39.7% | passed (NOISY; inconclusive) |

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
  "latency_case_processes_seconds": 28.925582542000484,
  "separate_cache_diagnostics_seconds": 0.36831645799975377,
  "collection_total_seconds": 29.2939005420003
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 25.64960345899999,
  "separate_cache_diagnostics_seconds": 0.5202376659999572,
  "collection_total_seconds": 26.169842166999842
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 69.996837917,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 5.33219825,
  "validation_metadata_and_collection_seconds": 64.533096833,
  "command_before_report_seconds": 69.865295083
}
```

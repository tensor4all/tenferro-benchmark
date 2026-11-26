# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261126_192058`.

### run_t1

```yaml
timestamp: '2026-11-26T19:20:59Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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
timestamp: '2026-11-26T19:21:38Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
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
- Expected 86, selected 86, executed 86, unsupported 0, failed 0, skipped 0, missing 0, noisy 14.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 86 | 86 | 86 | 0 | 0 | 0 | 0 | 14 |

- Noisy: `matmul_n8_count1024[pytorch]@t1`, `matmul_n8_count1024[faer]@t4`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t4`, `hadamard_f64_m64n64_dot_alloc[faer]@t4`, `hadamard_f64_m64n64_einsum_alloc[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`, `stream_f64_fresh_len32_einsum_alloc[faer]@t4`, `stream_f64_strides_len32_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.09 | 0.03 | 2041.02 | 2.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.10 | 0.12 | 3027.34 | 4.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 2.67 | 0.02 | 2607.42 | 0.4% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.05 | 0.26 | 2978.52 | 6.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.64 | 0.01 | 1601.56 | 0.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.86 | 0.06 | 1816.41 | 9.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.23 | 0.01 | 2177.73 | 0.3% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.86 | 0.12 | 1816.41 | 7.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 3.94 | 0.06 | 3847.66 | 1.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 7.20 | 0.31 | 7031.25 | 3.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 4.51 | 0.06 | 4404.30 | 0.9% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 10.36 | 0.20 | 10117.19 | 3.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.68 | 0.02 | 1640.62 | 1.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.77 | 0.06 | 1728.52 | 8.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.24 | 0.01 | 2187.50 | 0.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.75 | 0.06 | 1708.98 | 8.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 1.71 | 0.02 | 1669.92 | 1.0% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.05 | 0.14 | 2001.95 | 11.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.31 | 0.03 | 2255.86 | 18.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 1.94 | 0.17 | 1894.53 | 8.6% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 4.43 | 0.06 | 4326.17 | 1.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 20.87 | 0.28 | 20380.86 | 1.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 5.07 | 0.04 | 4951.17 | 0.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 20.91 | 0.71 | 20419.92 | 2.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.65 | 0.01 | 1611.33 | 0.4% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 16.41 | 0.14 | 16025.39 | 1.0% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.32 | 0.01 | 2265.62 | 0.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 16.22 | 0.21 | 15839.84 | 1.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 13.46 | 0.14 | 13144.53 | 1.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 37.75 | 0.41 | 36865.23 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 14.29 | 0.20 | 13955.08 | 1.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 52.68 | 0.58 | 51445.31 | 2.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.80 | 0.01 | 1757.81 | 0.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 16.84 | 0.32 | 16445.31 | 1.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.51 | 0.01 | 2451.17 | 0.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 16.93 | 0.10 | 16533.20 | 0.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.46 | 0.02 | 2402.34 | 0.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 17.36 | 0.20 | 16953.12 | 0.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 3.12 | 0.02 | 3046.88 | 0.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 17.67 | 0.23 | 17255.86 | 1.4% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 294967.00 | 878.17 | 0.6% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 2 | 299073.00 | 10183.75 | 10.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 165502.60 | 1262.20 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 4 | 167925.75 | 40891.25 | 11.8% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 221550.75 | 1097.00 | 0.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 3 | 225133.33 | 5997.83 | 9.0% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 57326.93 | 575.40 | 1.6% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 10 | 56774.90 | 31581.45 | 23.1% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 209445.75 | 2926.62 | 1.1% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 2 | 245813.50 | 9570.50 | 2.2% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 207750.00 | 678.75 | 0.5% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 240305.00 | 3196.00 | 1.7% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 46 | 15476.91 | 346.29 | 1.3% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 37 | 16427.92 | 4710.66 | 16.1% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 134 | 4627.81 | 143.56 | 1.7% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 88 | 7984.20 | 2238.43 | 19.0% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 6 | 77269.00 | 8892.42 | 8.2% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 12 | 67549.33 | 3227.29 | 6.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 17 | 52198.53 | 754.09 | 2.5% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 16 | 43982.06 | 5517.78 | 8.6% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 226860.50 | 2845.25 | 0.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 2 | 231106.00 | 11531.25 | 15.0% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 16 | 54215.50 | 594.56 | 4.9% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 10 | 57731.70 | 30973.80 | 22.2% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 21 | 28429.24 | 1689.33 | 3.5% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 17 | 42249.41 | 12257.29 | 17.4% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 121 | 2764.83 | 168.62 | 3.6% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 115 | 4586.64 | 365.65 | 12.0% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 57 | 11299.68 | 328.87 | 1.7% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 40 | 17495.88 | 5263.46 | 17.3% | passed (NOISY) |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.10 | 0.00 | 0.108 | 0.124 | 5.5% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.10 | 0.00 | 0.111 | 0.116 | 4.3% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.25 | 0.01 | 0.284 | 0.289 | 7.1% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.19 | 0.00 | 0.212 | 0.215 | 5.9% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.14 | 0.03 | 1.169 | 1.173 | 1.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 1.18 | 0.01 | 1.208 | 1.222 | 1.4% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 5.86 | 0.30 | 6.153 | 6.386 | 3.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.55 | 0.04 | 1.562 | 1.684 | 2.8% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.19 | 0.00 | 0.196 | 0.201 | 3.0% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.19 | 0.00 | 0.203 | 0.204 | 3.2% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.37 | 0.15 | 0.423 | 0.440 | 21.3% | passed (NOISY; inconclusive) |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.32 | 0.12 | 0.343 | 0.346 | 21.4% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.37 | 0.02 | 1.388 | 1.399 | 1.6% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.38 | 0.02 | 1.397 | 1.411 | 1.9% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 6.28 | 0.25 | 6.548 | 6.816 | 3.4% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 1.79 | 0.03 | 1.807 | 1.826 | 1.6% | passed |

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
  "latency_case_processes_seconds": 35.537504504000026,
  "separate_cache_diagnostics_seconds": 0.39319868100028543,
  "collection_total_seconds": 35.93070335599987
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 37.38742842600004,
  "separate_cache_diagnostics_seconds": 0.45198717800030863,
  "collection_total_seconds": 37.83941579400016
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 83.831066754,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 3.081062805,
  "validation_metadata_and_collection_seconds": 80.688448013,
  "command_before_report_seconds": 83.769510818
}
```

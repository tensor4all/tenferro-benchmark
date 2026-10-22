# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261022_184416`.

### run_t1

```yaml
timestamp: '2026-10-22T18:44:16Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 2007d101d23b5baf7deba4afb8c7b92da347dbba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 12e0b50f983ca3253ff46841eb15912448ae3e20
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-accelerate
    BENCHMARK_TARGET_PROFILE=mac-cpu /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 4
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

### run_t4

```yaml
timestamp: '2026-10-22T18:44:59Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 2007d101d23b5baf7deba4afb8c7b92da347dbba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 12e0b50f983ca3253ff46841eb15912448ae3e20
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-accelerate
    BENCHMARK_TARGET_PROFILE=mac-cpu /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 4
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

## Case status

- Manifest: `cpu/session_matrix` version 4, coverage `quick`, effort `standard`
- Expected 114, selected 114, executed 48, unsupported 66, failed 0, missing 0, noisy 30.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Unsupported: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n8_count1024[faer]@t4`, `matmul_n16_count1024[faer]@t4`, `matmul_n32_count1024[faer]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4` (+26 more)
- Noisy: `matmul_n2_count1024[blas]@t1`, `matmul_n2_count1024[pytorch]@t1`, `matmul_n4_count1024[blas]@t1`, `matmul_n4_count1024[pytorch]@t1`, `matmul_n8_count1024[blas]@t1`, `matmul_n8_count1024[pytorch]@t1`, `matmul_n16_count1024[blas]@t1`, `matmul_n16_count1024[pytorch]@t1`, `matmul_n32_count1024[blas]@t1`, `matmul_n32_count1024[pytorch]@t1`, `solve_n2_count1024[blas]@t1`, `solve_n2_count1024[pytorch]@t1`, `solve_n4_count1024[blas]@t1`, `solve_n4_count1024[pytorch]@t1`, `solve_n8_count1024[blas]@t1`, `solve_n8_count1024[pytorch]@t1`, `solve_n16_count1024[blas]@t1`, `solve_n16_count1024[pytorch]@t1`, `solve_n32_count1024[blas]@t1`, `solve_n32_count1024[pytorch]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[blas]@t1`, `matmul_n2_count1024[blas]@t4`, `matmul_n2_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `matmul_n32_count1024[blas]@t4`, `matmul_n32_count1024[pytorch]@t4`, `solve_n8_count1024[pytorch]@t4`, `solve_n16_count1024[blas]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | accelerate | shared-session | not applicable | 1.48 | 0.31 | 1445.31 | 45.1% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.49 | 0.36 | 1455.08 | 28.1% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | accelerate | shared-session | not applicable | 1.31 | 0.02 | 1279.30 | 2.2% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.14 | 0.22 | 1113.28 | 25.9% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | accelerate | shared-session | not applicable | 0.95 | 0.14 | 927.73 | 10.5% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.75 | 0.19 | 732.42 | 34.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | accelerate | shared-session | not applicable | 0.91 | 0.14 | 888.67 | 23.7% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.79 | 0.09 | 771.48 | 13.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | accelerate | shared-session | not applicable | 2.30 | 1.07 | 2246.09 | 36.8% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 2.55 | 1.43 | 2490.23 | 52.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | accelerate | shared-session | not applicable | 2.52 | 0.68 | 2460.94 | 28.0% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.83 | 0.25 | 1787.11 | 13.9% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | accelerate | shared-session | not applicable | 1.28 | 1.07 | 1250.00 | 48.6% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.75 | 0.12 | 732.42 | 30.8% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | accelerate | shared-session | not applicable | 0.93 | 0.04 | 908.20 | 3.1% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.79 | 0.04 | 771.48 | 5.8% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | accelerate | shared-session | not applicable | 1.26 | 0.31 | 1230.47 | 13.4% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.92 | 0.15 | 898.44 | 19.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | accelerate | shared-session | not applicable | 1.00 | 0.04 | 976.56 | 6.1% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.89 | 0.04 | 869.14 | 6.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | accelerate | shared-session | not applicable | 3.36 | 2.11 | 3281.25 | 38.1% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 13.68 | 2.98 | 13359.38 | 19.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | accelerate | shared-session | not applicable | 3.30 | 0.60 | 3222.66 | 31.1% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 10.61 | 0.22 | 10361.33 | 7.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | accelerate | shared-session | not applicable | 0.90 | 0.46 | 878.91 | 40.8% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 13.51 | 3.36 | 13193.36 | 22.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | accelerate | shared-session | not applicable | 0.83 | 0.03 | 810.55 | 3.4% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 7.66 | 0.71 | 7480.47 | 8.2% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | accelerate | shared-session | not applicable | 13.09 | 4.27 | 12783.20 | 21.5% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 21.93 | 5.53 | 21416.02 | 17.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | accelerate | shared-session | not applicable | 8.27 | 0.21 | 8076.17 | 2.0% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 16.86 | 1.05 | 16464.84 | 6.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | accelerate | shared-session | not applicable | 1.64 | 0.37 | 1601.56 | 18.1% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 11.79 | 2.07 | 11513.67 | 21.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | accelerate | shared-session | not applicable | 0.99 | 0.08 | 966.80 | 6.7% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 9.33 | 0.33 | 9111.33 | 8.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | accelerate | shared-session | not applicable | 1.54 | 0.67 | 1503.91 | 31.7% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 15.38 | 3.53 | 15019.53 | 22.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | accelerate | shared-session | not applicable | 1.33 | 0.09 | 1298.83 | 6.3% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 14.71 | 4.91 | 14365.23 | 24.3% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | 128 | 0.75 | 0.06 | 0.825 | 0.857 | 5.7% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | 128 | 0.83 | 0.12 | 0.966 | 1.157 | 12.1% | passed (NOISY; inconclusive) |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | 128 | 4.17 | 0.61 | 5.036 | 5.832 | 13.0% | passed (NOISY; inconclusive) |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | 128 | 1.54 | 0.24 | 2.084 | 3.271 | 29.8% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | 128 | 0.75 | 0.05 | 0.808 | 0.838 | 5.1% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | 128 | 0.72 | 0.01 | 0.762 | 0.766 | 2.4% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | 128 | 3.14 | 0.12 | 3.408 | 3.523 | 4.8% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | 128 | 1.25 | 0.02 | 1.275 | 1.290 | 1.3% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |

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
  "latency_case_processes_seconds": 38.214109040999574,
  "separate_cache_diagnostics_seconds": 0.49788975000046776,
  "collection_total_seconds": 38.71199912500015
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 27.063230582999495,
  "separate_cache_diagnostics_seconds": 0.2717116250005347,
  "collection_total_seconds": 27.334942542000135
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 80.271546459,
  "report_generation_seconds": 0.122072875,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 3.938312542,
  "validation_metadata_and_collection_seconds": 76.211161042,
  "command_before_report_seconds": 80.149473584
}
```

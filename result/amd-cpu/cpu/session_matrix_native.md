# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261013_102936`.

### run_t1

```yaml
timestamp: '2026-10-13T10:29:36Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 17803e02ebacc89da44664078bbb54092e141d20
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=600 BENCH_EFFORT=standard BENCH_COVERAGE=full
    TENFERRO_CPU_FEATURES=native BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: full
  manifest_version: 4
  selection_filter: null
  expected_cases: 74
  selected_cases: 74
```

### run_t4

```yaml
timestamp: '2026-10-13T10:30:33Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - native
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 17803e02ebacc89da44664078bbb54092e141d20
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=600 BENCH_EFFORT=standard BENCH_COVERAGE=full
    TENFERRO_CPU_FEATURES=native BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: full
  manifest_version: 4
  selection_filter: null
  expected_cases: 74
  selected_cases: 74
```

## Case status

- Manifest: `cpu/session_matrix` version 4, coverage `full`, effort `standard`
- Expected 200, selected 200, executed 136, unsupported 64, failed 0, missing 0, noisy 11.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Unsupported: `matmul_n2_count1024[blas]@t1`, `matmul_n4_count1024[blas]@t1`, `matmul_n8_count1024[blas]@t1`, `matmul_n16_count1024[blas]@t1`, `matmul_n32_count1024[blas]@t1`, `solve_n2_count1024[blas]@t1`, `solve_n4_count1024[blas]@t1`, `solve_n8_count1024[blas]@t1`, `solve_n16_count1024[blas]@t1`, `solve_n32_count1024[blas]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_backend-outer-min-items-4096[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_scoped-outer-min-items-4096[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-sequential[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-sequential[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-provider-items[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-provider-items[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_backend-lane-min-work-0[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_backend-lane-min-work-0[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_scoped-lane-min-work-0[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_scoped-lane-min-work-0[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_thrash_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_cold_len128_einsum3_alloc[blas]@t1`, `matmul_n2_count1024[blas]@t4`, `matmul_n4_count1024[blas]@t4`, `matmul_n8_count1024[blas]@t4`, `matmul_n16_count1024[blas]@t4`, `matmul_n32_count1024[blas]@t4`, `solve_n2_count1024[blas]@t4`, `solve_n4_count1024[blas]@t4`, `solve_n8_count1024[blas]@t4` (+24 more)
- Noisy: `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t4`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t4`, `bdot_f64_b64_m32n32k32_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m32n32k32_direct_into_auto[faer]@t4`, `hadamard_f64_m256n256_dot_alloc[faer]@t4`, `hadamard_f64_m256n256_einsum_alloc[faer]@t4`, `chain3_f64_n64_einsum_alloc[faer]@t4`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t4`, `eager_stream_f64_thrash_len128_einsum3_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 8.94 | 0.23 | 8730.47 | 2.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.02 | 0.16 | 2949.22 | 3.6% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 9.01 | 0.33 | 8798.83 | 2.7% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 2.88 | 0.15 | 2812.50 | 5.0% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 7.42 | 0.38 | 7246.09 | 2.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 2.09 | 0.05 | 2041.02 | 6.1% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 7.66 | 0.33 | 7480.47 | 19.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 2.16 | 0.09 | 2109.38 | 4.3% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 13.15 | 0.16 | 12841.80 | 1.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 4.93 | 0.14 | 4814.45 | 3.6% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 12.90 | 0.26 | 12597.66 | 1.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 5.58 | 0.28 | 5449.22 | 4.3% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 7.53 | 0.16 | 7353.52 | 1.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 2.08 | 0.13 | 2031.25 | 5.2% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 7.49 | 0.18 | 7314.45 | 2.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 2.14 | 0.23 | 2089.84 | 8.2% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 8.12 | 0.18 | 7929.69 | 2.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.49 | 0.14 | 2431.64 | 4.9% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 8.16 | 0.31 | 7968.75 | 3.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.26 | 0.06 | 2207.03 | 5.4% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 5.76 | 0.09 | 5625.00 | 1.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 23.64 | 0.41 | 23085.94 | 1.1% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 116.95 | 1.37 | 114208.98 | 1.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 24.21 | 1.18 | 23642.58 | 2.8% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 1.26 | 0.06 | 1230.47 | 2.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 18.68 | 0.38 | 18242.19 | 1.7% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 101.65 | 1.56 | 99267.58 | 4.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 18.82 | 0.25 | 18378.91 | 1.7% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 16.63 | 0.20 | 16240.23 | 2.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 36.52 | 0.96 | 35664.06 | 1.5% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 136.42 | 1.20 | 133222.66 | 1.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 36.27 | 0.84 | 35419.92 | 1.8% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 1.49 | 0.03 | 1455.08 | 1.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 18.45 | 0.29 | 18017.58 | 1.8% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 103.28 | 2.09 | 100859.38 | 1.6% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 19.27 | 0.57 | 18818.36 | 2.4% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.25 | 0.05 | 2197.27 | 2.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 20.13 | 0.39 | 19658.20 | 1.9% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 106.21 | 2.23 | 103720.70 | 1.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 19.35 | 0.31 | 18896.48 | 1.7% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_c64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | c64 | 1024×4×4×4 | 1 | 1 | 5 | 164844.20 | 2362.20 | 1.6% | passed |
| bdot_c64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | c64 | 1024×4×4×4 | 4 | 4 | 5 | 168178.00 | 10057.60 | 6.0% | passed |
| bdot_c64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | c64 | 1024×4×4×4 | 1 | 1 | 11 | 60720.82 | 2530.68 | 2.7% | passed |
| bdot_c64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | c64 | 1024×4×4×4 | 4 | 4 | 13 | 63745.31 | 1513.50 | 3.0% | passed |
| bdot_c64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | c64 | 1024×4×4×4 | 1 | 1 | 5 | 168802.80 | 6139.50 | 3.0% | passed |
| bdot_c64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | c64 | 1024×4×4×4 | 4 | 4 | 5 | 172320.40 | 5951.40 | 3.8% | passed |
| bdot_c64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | c64 | 1024×4×4×4 | 1 | 1 | 12 | 62646.25 | 7502.87 | 7.3% | passed |
| bdot_c64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | c64 | 1024×4×4×4 | 4 | 4 | 11 | 61390.18 | 3035.55 | 4.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 154024.00 | 9589.50 | 3.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 6 | 154793.00 | 4722.83 | 2.3% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 8 | 110681.38 | 3794.56 | 3.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 8 | 113320.12 | 5355.88 | 4.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 152673.40 | 2574.70 | 2.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 4 | 153205.75 | 7793.00 | 5.1% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_backend-outer-min-items-4096 | concrete / allocating / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_backend-outer-min-items-4096 | concrete / allocating / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel | concrete / allocating / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel | concrete / allocating / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-provider-items | concrete / allocating / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-provider-items | concrete / allocating / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-sequential | concrete / allocating / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-sequential | concrete / allocating / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor | concrete / allocating / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor | concrete / allocating / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096 | concrete / allocating / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096 | concrete / allocating / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 49222.93 | 2043.77 | 3.8% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 15 | 48587.33 | 1534.83 | 3.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096 | concrete / reused / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096 | concrete / reused / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel | concrete / reused / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel | concrete / reused / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-provider-items | concrete / reused / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-provider-items | concrete / reused / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-sequential | concrete / reused / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-sequential | concrete / reused / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor | concrete / reused / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor | concrete / reused / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_scoped-outer-min-items-4096 | concrete / reused / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_direct_into_scoped-outer-min-items-4096 | concrete / reused / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 156617.60 | 5689.30 | 5.0% | passed |
| bdot_f64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 5 | 152776.40 | 3608.90 | 4.7% | passed |
| bdot_f64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 16 | 49834.19 | 4405.28 | 5.9% | passed |
| bdot_f64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 15 | 49010.47 | 2327.10 | 4.9% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 2 | 242365.50 | 16905.00 | 4.8% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 2 | 303099.50 | 41390.75 | 14.8% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 165323.67 | 16128.00 | 8.2% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 2 | 296963.00 | 47737.50 | 20.4% | passed (NOISY) |
| bdot_f64_b256_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 256×4×4×4 | 1 | 1 | 17 | 45250.53 | 1991.47 | 4.8% | passed |
| bdot_f64_b256_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 256×4×4×4 | 4 | 4 | 18 | 45419.33 | 1240.31 | 3.5% | passed |
| bdot_f64_b256_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 256×4×4×4 | 1 | 1 | 37 | 18579.68 | 940.20 | 5.2% | passed |
| bdot_f64_b256_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 256×4×4×4 | 4 | 4 | 34 | 18164.56 | 824.65 | 3.0% | passed |
| bdot_f64_b320_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 320×4×4×4 | 1 | 1 | 15 | 54174.67 | 2270.17 | 4.1% | passed |
| bdot_f64_b320_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 320×4×4×4 | 4 | 4 | 15 | 54081.80 | 2289.90 | 4.3% | passed |
| bdot_f64_b320_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 320×4×4×4 | 1 | 1 | 36 | 20088.75 | 874.31 | 5.3% | passed |
| bdot_f64_b320_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 320×4×4×4 | 4 | 4 | 34 | 20529.59 | 614.29 | 4.7% | passed |
| bdot_f64_b4096_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 4096×4×4×4 | 1 | 1 | 1 | 613433.00 | 44601.00 | 5.6% | passed |
| bdot_f64_b4096_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 4096×4×4×4 | 4 | 4 | 1 | 602826.00 | 26484.50 | 2.7% | passed |
| bdot_f64_b4096_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 4096×4×4×4 | 1 | 1 | 4 | 168610.75 | 9813.88 | 5.0% | passed |
| bdot_f64_b4096_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 4096×4×4×4 | 4 | 4 | 5 | 168648.40 | 3862.20 | 2.8% | passed |
| bdot_f64_b64_m16n16k16_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×16×16×16 | 1 | 1 | 13 | 52030.77 | 3132.12 | 4.6% | passed |
| bdot_f64_b64_m16n16k16_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×16×16×16 | 4 | 4 | 11 | 51519.73 | 4503.68 | 8.1% | passed |
| bdot_f64_b64_m16n16k16_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×16×16×16 | 1 | 1 | 22 | 25097.91 | 1046.32 | 4.0% | passed |
| bdot_f64_b64_m16n16k16_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×16×16×16 | 4 | 4 | 19 | 25070.11 | 1718.89 | 7.5% | passed |
| bdot_f64_b64_m32n32k32_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×32×32×32 | 1 | 1 | 3 | 193580.00 | 8849.67 | 3.9% | passed |
| bdot_f64_b64_m32n32k32_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×32×32×32 | 4 | 4 | 2 | 361890.00 | 53720.75 | 11.9% | passed (NOISY) |
| bdot_f64_b64_m32n32k32_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×32×32×32 | 1 | 1 | 5 | 103546.80 | 4755.90 | 3.3% | passed |
| bdot_f64_b64_m32n32k32_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×32×32×32 | 4 | 4 | 2 | 260585.00 | 51160.50 | 14.4% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 37 | 17318.70 | 1075.81 | 5.6% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 37 | 17367.70 | 746.74 | 3.9% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_backend-lane-min-work-0 | concrete / allocating / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_alloc_backend-lane-min-work-0 | concrete / allocating / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_alloc_scoped-lane-min-work-0 | concrete / allocating / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_alloc_scoped-lane-min-work-0 | concrete / allocating / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 53 | 10049.45 | 630.12 | 4.0% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 54 | 9941.30 | 881.35 | 5.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_backend-lane-min-work-0 | concrete / reused / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_into_backend-lane-min-work-0 | concrete / reused / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_into_scoped-lane-min-work-0 | concrete / reused / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m4n4k4_direct_into_scoped-lane-min-work-0 | concrete / reused / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | — | — | — | — | — | unsupported: the batch-policy API was removed in tenferro-rs #2004; only the compiled backend's default policy is |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 4 | 200906.25 | 41047.62 | 10.6% | passed (NOISY) |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 4 | 181549.75 | 8539.50 | 4.1% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 8 | 60089.25 | 4215.25 | 7.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 8 | 59239.75 | 5251.81 | 9.7% | passed |
| bdot_f64_b64_m8n8k8_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×8×8×8 | 1 | 1 | 30 | 23800.43 | 1094.10 | 3.3% | passed |
| bdot_f64_b64_m8n8k8_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×8×8×8 | 4 | 4 | 16 | 23030.94 | 1605.62 | 7.7% | passed |
| bdot_f64_b64_m8n8k8_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×8×8×8 | 1 | 1 | 45 | 11731.22 | 798.60 | 4.8% | passed |
| bdot_f64_b64_m8n8k8_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×8×8×8 | 4 | 4 | 50 | 11868.18 | 543.29 | 2.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 172782.60 | 11726.70 | 4.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 5 | 170217.00 | 5163.20 | 3.2% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 49606.27 | 1932.03 | 4.6% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 47860.62 | 1163.03 | 4.5% | passed |
| beinsum_f64_b1024_m4n4k4_strided_alloc_auto | concrete-einsum / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 169816.60 | 3975.80 | 3.5% | passed |
| beinsum_f64_b1024_m4n4k4_strided_alloc_auto | concrete-einsum / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 5 | 173157.40 | 13910.70 | 6.4% | passed |
| beinsum_f64_b1024_m4n4k4_strided_into_auto | concrete-einsum / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 11 | 48700.64 | 3960.09 | 5.0% | passed |
| beinsum_f64_b1024_m4n4k4_strided_into_auto | concrete-einsum / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 15 | 49722.13 | 2795.87 | 4.1% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 24 | 29506.21 | 1083.79 | 3.8% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 22 | 29504.18 | 1256.75 | 3.6% | passed |
| chain3_f64_n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×64 | 1 | 1 | 10 | 52235.70 | 3606.90 | 5.9% | passed |
| chain3_f64_n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×64 | 4 | 4 | 10 | 52883.60 | 7519.95 | 13.9% | passed (NOISY) |
| hadamard_f64_m256n256_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×256×256×1 | 1 | 1 | 13 | 29844.54 | 1745.58 | 6.8% | passed |
| hadamard_f64_m256n256_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×256×256×1 | 4 | 4 | 2 | 134204.50 | 278610.75 | 85.8% | passed (NOISY) |
| hadamard_f64_m256n256_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×256×256×1 | 1 | 1 | 11 | 38958.55 | 3223.91 | 9.7% | passed |
| hadamard_f64_m256n256_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×256×256×1 | 4 | 4 | 3 | 197386.67 | 18286.17 | 12.0% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 86 | 4734.99 | 340.88 | 5.7% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 80 | 4702.14 | 270.82 | 4.0% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 46 | 13486.30 | 916.11 | 5.4% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 7 | 93477.57 | 12863.57 | 7.9% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.74 | 0.03 | 0.756 | 0.759 | 2.7% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.74 | 0.03 | 0.774 | 0.866 | 5.0% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.76 | 0.02 | 0.791 | 0.878 | 4.7% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.75 | 0.05 | 0.840 | 0.870 | 6.0% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 3.50 | 0.14 | 3.588 | 3.890 | 3.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 3.75 | 0.18 | 3.889 | 4.042 | 3.4% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 8.78 | 0.33 | 9.119 | 9.149 | 2.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 3.54 | 0.19 | 3.732 | 3.939 | 4.0% | passed |
| eager_stream_f64_thrash_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_thrash_len128_einsum3_alloc | faer | 1 | 128 | 8.62 | 0.21 | 8.788 | 8.893 | 1.9% | passed |
| eager_stream_f64_cold_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_cold_len128_einsum3_alloc | faer | 1 | 128 | 3.62 | 0.17 | 3.772 | 3.840 | 3.4% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.76 | 0.01 | 0.781 | 0.866 | 4.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.76 | 0.03 | 0.794 | 0.852 | 4.5% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.76 | 0.01 | 0.791 | 0.878 | 4.7% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.77 | 0.02 | 0.819 | 0.872 | 4.7% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 3.54 | 0.16 | 3.677 | 3.680 | 3.1% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 3.51 | 0.09 | 3.708 | 3.764 | 3.0% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 9.45 | 2.76 | 12.509 | 12.921 | 15.7% | passed (NOISY; inconclusive) |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 3.52 | 0.18 | 3.681 | 3.702 | 3.1% | passed |
| eager_stream_f64_thrash_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_thrash_len128_einsum3_alloc | faer | 4 | 128 | 8.30 | 0.25 | 10.110 | 18.590 | 29.0% | passed (NOISY; inconclusive) |
| eager_stream_f64_cold_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_cold_len128_einsum3_alloc | faer | 4 | 128 | 3.49 | 0.09 | 3.596 | 3.807 | 2.8% | passed |

### Separate-process cache observations

Public EagerRuntime extension-store counters only. Provider GEMM-analysis/indexed-plan caches and runtime prepared caches are outside this scope. No public plan-build counter exists; lookup misses are not treated as builds. These processes collect no latency samples. Hits/misses/evictions are measured-pass deltas; retained bytes are a logical estimate, not process RSS.

| Case | Threads | Hits | Misses | Evictions | Final entries | Final retained bytes | Builds | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| eager_stream_f64_fixed_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 4 | 5408 | unavailable | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | 1 | 0 | 1920 | 1920 | 256 | 346112 | unavailable | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | 1 | 1920 | 0 | 0 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_thrash_len128_einsum3_alloc | 1 | 0 | 1920 | 1920 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_cold_len128_einsum3_alloc | 1 | 1905 | 15 | 0 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 4 | 5408 | unavailable | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | 4 | 0 | 1920 | 1920 | 256 | 346112 | unavailable | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | 4 | 1920 | 0 | 0 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_thrash_len128_einsum3_alloc | 4 | 0 | 1920 | 1920 | 1 | 1352 | unavailable | passed |
| eager_stream_f64_cold_len128_einsum3_alloc | 4 | 1905 | 15 | 0 | 1 | 1352 | unavailable | passed |

Wall observations `wall_collection_t1.json`:
```json
{
  "latency_case_processes_seconds": 51.11899272099981,
  "separate_cache_diagnostics_seconds": 1.0653784430005544,
  "collection_total_seconds": 52.184371446000114
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 60.78742674499972,
  "separate_cache_diagnostics_seconds": 1.1124166680001508,
  "collection_total_seconds": 61.899843580999914
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 126.371366547,
  "report_generation_seconds": 0.173497429,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 600.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 0.795307845,
  "validation_metadata_and_collection_seconds": 125.402561273,
  "command_before_report_seconds": 126.197869118
}
```

# CPU shared-session matrix results

Raw data and provenance: `data/results/zen5-cpu/cpu/session_matrix/20261028_011306`.

### run_t1

```yaml
timestamp: '2026-10-28T01:13:06Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 3f31f38e0491084b9414035754a3809ef0f5ae94
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=zen5-cpu
    /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh 1 4 8 '
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
timestamp: '2026-10-28T01:13:27Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 3f31f38e0491084b9414035754a3809ef0f5ae94
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=zen5-cpu
    /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh 1 4 8 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 4
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

### run_t8

```yaml
timestamp: '2026-10-28T01:13:48Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8fec1ca70852b930c092e7177c30bb8af7594a57
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 3f31f38e0491084b9414035754a3809ef0f5ae94
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=zen5-cpu
    /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh 1 4 8 '
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
- Expected 171, selected 171, executed 72, unsupported 99, failed 0, missing 0, noisy 7.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Unsupported: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n8_count1024[faer]@t4`, `matmul_n16_count1024[faer]@t4`, `matmul_n32_count1024[faer]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4` (+59 more)
- Noisy: `matmul_n8_count1024[pytorch]@t1`, `matmul_n16_count1024[pytorch]@t1`, `matmul_n4_count1024[pytorch]@t4`, `matmul_n8_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `matmul_n32_count1024[blas]@t4`, `matmul_n8_count1024[pytorch]@t8`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | blas | shared-session | not applicable | 1.15 | 0.01 | 1123.05 | 2.0% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.30 | 0.03 | 1269.53 | 12.3% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | blas | shared-session | not applicable | 1.30 | 0.01 | 1269.53 | 0.7% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.30 | 0.03 | 1269.53 | 12.6% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | blas | shared-session | not applicable | 2.25 | 0.02 | 2197.27 | 0.7% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 2.39 | 0.05 | 2333.98 | 8.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | blas | shared-session | not applicable | 0.73 | 0.00 | 712.89 | 0.3% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.79 | 0.01 | 771.48 | 8.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | blas | shared-session | not applicable | 0.90 | 0.01 | 878.91 | 1.0% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.81 | 0.02 | 791.02 | 6.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | blas | shared-session | not applicable | 1.38 | 0.01 | 1347.66 | 0.3% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 1.23 | 0.01 | 1201.17 | 8.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | blas | shared-session | not applicable | 4.02 | 0.04 | 3925.78 | 0.8% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 4.59 | 0.20 | 4482.42 | 4.2% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | blas | shared-session | not applicable | 4.04 | 1.98 | 3945.31 | 96.2% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 4.62 | 0.15 | 4511.72 | 3.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | blas | shared-session | not applicable | 5.25 | 0.02 | 5126.95 | 0.4% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 5.77 | 0.07 | 5634.77 | 2.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | blas | shared-session | not applicable | 0.75 | 0.00 | 732.42 | 1.2% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.75 | 0.01 | 732.42 | 9.9% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | blas | shared-session | not applicable | 0.91 | 0.01 | 888.67 | 1.0% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.75 | 0.01 | 732.42 | 10.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | blas | shared-session | not applicable | 1.41 | 0.01 | 1376.95 | 0.5% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 1.20 | 0.00 | 1171.88 | 9.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | blas | shared-session | not applicable | 0.83 | 0.00 | 810.55 | 0.2% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.81 | 0.01 | 791.02 | 13.8% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | blas | shared-session | not applicable | 0.95 | 0.02 | 927.73 | 1.2% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.85 | 0.02 | 830.08 | 12.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | blas | shared-session | not applicable | 1.47 | 0.00 | 1435.55 | 0.2% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 1.34 | 0.13 | 1308.59 | 13.7% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | blas | shared-session | not applicable | 2.12 | 0.20 | 2070.31 | 4.9% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 8.66 | 0.31 | 8457.03 | 1.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | blas | shared-session | not applicable | 2.32 | 0.21 | 2265.62 | 4.5% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 8.78 | 0.08 | 8574.22 | 2.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | blas | shared-session | not applicable | 3.77 | 0.35 | 3681.64 | 4.7% | passed |
| solve_n16_count1024 | solve | n16 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 13.54 | 0.37 | 13222.66 | 1.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | blas | shared-session | not applicable | 0.61 | 0.00 | 595.70 | 1.7% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 6.61 | 0.06 | 6455.08 | 1.2% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | blas | shared-session | not applicable | 0.74 | 0.01 | 722.66 | 1.1% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 6.54 | 0.03 | 6386.72 | 1.1% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | blas | shared-session | not applicable | 1.19 | 0.01 | 1162.11 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 10.29 | 0.05 | 10048.83 | 1.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | blas | shared-session | not applicable | 9.33 | 1.19 | 9111.33 | 6.2% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 15.87 | 0.09 | 15498.05 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | blas | shared-session | not applicable | 9.23 | 1.34 | 9013.67 | 7.2% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 15.21 | 0.05 | 14853.52 | 0.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | blas | shared-session | not applicable | 13.48 | 1.84 | 13164.06 | 6.7% | passed |
| solve_n32_count1024 | solve | n32 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 23.05 | 0.19 | 22509.77 | 0.6% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | blas | shared-session | not applicable | 0.66 | 0.01 | 644.53 | 0.6% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 6.73 | 0.05 | 6572.27 | 1.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | blas | shared-session | not applicable | 0.79 | 0.01 | 771.48 | 1.0% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 6.71 | 0.09 | 6552.73 | 1.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | blas | shared-session | not applicable | 1.24 | 0.00 | 1210.94 | 0.2% | passed |
| solve_n4_count1024 | solve | n4 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 10.73 | 0.17 | 10478.52 | 1.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | blas | shared-session | not applicable | 1.19 | 0.00 | 1162.11 | 0.3% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 7.03 | 0.02 | 6865.23 | 0.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | blas | shared-session | not applicable | 1.33 | 0.01 | 1298.83 | 1.2% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 7.20 | 0.04 | 7031.25 | 0.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | blas | shared-session | not applicable | 2.06 | 0.01 | 2011.72 | 0.3% | passed |
| solve_n8_count1024 | solve | n8 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 11.11 | 0.05 | 10849.61 | 1.0% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 1 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 4 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | None | None×None×None×None | 8 | — | — | — | — | — | unsupported: harness compiled with the blas CPU backend; this case is declared for the faer provider |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | 128 | 0.76 | 0.03 | 0.787 | 0.807 | 3.3% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | 128 | 0.73 | 0.01 | 0.745 | 0.761 | 2.5% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | 128 | 2.49 | 0.05 | 2.563 | 2.621 | 1.9% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | 128 | 1.02 | 0.02 | 1.036 | 1.043 | 1.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | 128 | 0.82 | 0.01 | 0.844 | 0.853 | 2.1% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | 128 | 0.82 | 0.01 | 0.834 | 0.847 | 1.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | 128 | 2.63 | 0.07 | 2.715 | 2.764 | 1.9% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | 128 | 1.11 | 0.01 | 1.130 | 1.137 | 1.2% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 8 | 128 | 1.10 | 0.02 | 1.129 | 1.135 | 1.2% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 8 | 128 | 1.12 | 0.01 | 1.139 | 1.153 | 1.2% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 8 | 128 | 3.79 | 0.11 | 3.900 | 4.016 | 2.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 8 | 128 | 1.55 | 0.02 | 1.563 | 1.576 | 0.7% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |

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
| eager_stream_f64_fixed_len128_einsum3_alloc | 8 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | 8 | 1920 | 0 | 0 | 4 | 5792 | unavailable | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | 8 | 0 | 1920 | 1920 | 256 | 370688 | unavailable | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | 8 | 1920 | 0 | 0 | 1 | 1448 | unavailable | passed |

Wall observations `wall_collection_t1.json`:
```json
{
  "latency_case_processes_seconds": 17.45642616999976,
  "separate_cache_diagnostics_seconds": 0.28663118799886433,
  "collection_total_seconds": 17.743057617997692
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 17.824351237002702,
  "separate_cache_diagnostics_seconds": 0.2990248120040633,
  "collection_total_seconds": 18.123376259005454
}
```

Wall observations `wall_collection_t8.json`:
```json
{
  "latency_case_processes_seconds": 26.58458687699749,
  "separate_cache_diagnostics_seconds": 0.4175302229996305,
  "collection_total_seconds": 27.002117299998645
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_before_final_report_write": 73.383985056,
  "report_generation_seconds": 0.102034821,
  "scope": "command startup through report generation; excludes final report file writes",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 0.154756526,
  "validation_metadata_and_collection_seconds": 73.127193709,
  "command_before_report_seconds": 73.281950235
}
```

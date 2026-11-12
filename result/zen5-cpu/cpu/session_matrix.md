# CPU shared-session matrix results

Raw data and provenance: `data/results/zen5-cpu/cpu/session_matrix/20261112_155237`.

### run_t1

```yaml
timestamp: '2026-11-12T15:52:37Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: 918046a6065be464f13b727654679c3a00650c06
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=native BENCHMARK_TARGET_PROFILE=zen5-cpu
    scripts/run_cpu_session.sh 1 4 8 '
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
timestamp: '2026-11-12T15:53:20Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: 918046a6065be464f13b727654679c3a00650c06
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=native BENCHMARK_TARGET_PROFILE=zen5-cpu
    scripts/run_cpu_session.sh 1 4 8 '
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
timestamp: '2026-11-12T15:54:03Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: 918046a6065be464f13b727654679c3a00650c06
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=native BENCHMARK_TARGET_PROFILE=zen5-cpu
    scripts/run_cpu_session.sh 1 4 8 '
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
- Expected 171, selected 171, executed 129, unsupported 42, failed 0, skipped 0, missing 0, noisy 7.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 171 | 171 | 129 | 42 | 0 | 0 | 0 | 7 |

- Unsupported: `matmul_n2_count1024[blas]@t1`, `matmul_n4_count1024[blas]@t1`, `matmul_n8_count1024[blas]@t1`, `matmul_n16_count1024[blas]@t1`, `matmul_n32_count1024[blas]@t1`, `solve_n2_count1024[blas]@t1`, `solve_n4_count1024[blas]@t1`, `solve_n8_count1024[blas]@t1`, `solve_n16_count1024[blas]@t1`, `solve_n32_count1024[blas]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[blas]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[blas]@t1`, `matmul_n2_count1024[blas]@t4`, `matmul_n4_count1024[blas]@t4`, `matmul_n8_count1024[blas]@t4`, `matmul_n16_count1024[blas]@t4`, `matmul_n32_count1024[blas]@t4`, `solve_n2_count1024[blas]@t4`, `solve_n4_count1024[blas]@t4`, `solve_n8_count1024[blas]@t4`, `solve_n16_count1024[blas]@t4`, `solve_n32_count1024[blas]@t4`, `eager_stream_f64_fixed_len128_einsum3_alloc[blas]@t4`, `eager_stream_f64_mixed_len128_einsum3_alloc[blas]@t4`, `eager_stream_f64_fresh_len128_einsum3_alloc[blas]@t4`, `eager_stream_f64_strides_len128_einsum3_alloc[blas]@t4`, `matmul_n2_count1024[blas]@t8`, `matmul_n4_count1024[blas]@t8`, `matmul_n8_count1024[blas]@t8`, `matmul_n16_count1024[blas]@t8`, `matmul_n32_count1024[blas]@t8`, `solve_n2_count1024[blas]@t8`, `solve_n4_count1024[blas]@t8`, `solve_n8_count1024[blas]@t8`, `solve_n16_count1024[blas]@t8`, `solve_n32_count1024[blas]@t8`, `eager_stream_f64_fixed_len128_einsum3_alloc[blas]@t8`, `eager_stream_f64_mixed_len128_einsum3_alloc[blas]@t8` (+2 more)
- Noisy: `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t4`, `hadamard_f64_m64n64_dot_alloc[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t8`, `stream_f64_strides_len32_einsum_alloc[faer]@t8`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.41 | 0.01 | 2353.52 | 1.3% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.12 | 0.05 | 3046.88 | 5.0% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 2.86 | 0.01 | 2792.97 | 0.7% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.13 | 0.09 | 3056.64 | 5.0% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | faer | shared-session | not applicable | 3.07 | 0.06 | 2998.05 | 1.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 3.78 | 0.06 | 3691.41 | 5.5% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 2.10 | 0.00 | 2050.78 | 1.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 2.00 | 0.02 | 1953.12 | 6.6% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.46 | 0.00 | 2402.34 | 0.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 2.02 | 0.01 | 1972.66 | 6.8% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | faer | shared-session | not applicable | 2.57 | 0.01 | 2509.77 | 0.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 2.05 | 0.05 | 2001.95 | 8.0% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 5.28 | 0.06 | 5156.25 | 1.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 8.54 | 0.14 | 8339.84 | 2.3% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 6.15 | 0.06 | 6005.86 | 1.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 8.24 | 0.12 | 8046.88 | 2.5% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | faer | shared-session | not applicable | 6.35 | 0.03 | 6201.17 | 0.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 8.19 | 0.10 | 7998.05 | 2.4% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 2.16 | 0.00 | 2109.38 | 0.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.93 | 0.01 | 1884.77 | 6.1% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.50 | 0.00 | 2441.41 | 0.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.92 | 0.01 | 1875.00 | 5.9% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | faer | shared-session | not applicable | 2.56 | 0.01 | 2500.00 | 0.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 1.95 | 0.03 | 1904.30 | 8.6% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.13 | 0.01 | 2080.08 | 0.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.02 | 0.05 | 1972.66 | 6.6% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.50 | 0.01 | 2441.41 | 0.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.04 | 0.07 | 1992.19 | 8.6% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | faer | shared-session | not applicable | 2.58 | 0.01 | 2519.53 | 0.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 2.31 | 0.16 | 2255.86 | 7.0% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 5.25 | 0.03 | 5126.95 | 0.5% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 21.27 | 0.34 | 20771.48 | 1.4% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 5.76 | 0.02 | 5625.00 | 0.3% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 21.83 | 0.11 | 21318.36 | 0.8% | passed |
| solve_n16_count1024 | solve | n16 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | faer | shared-session | not applicable | 6.20 | 0.07 | 6054.69 | 1.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 23.34 | 0.14 | 22792.97 | 0.8% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 2.14 | 0.01 | 2089.84 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 17.20 | 0.05 | 16796.88 | 0.7% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.52 | 0.01 | 2460.94 | 1.0% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 17.61 | 0.14 | 17197.27 | 0.9% | passed |
| solve_n2_count1024 | solve | n2 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | faer | shared-session | not applicable | 2.66 | 0.01 | 2597.66 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 17.20 | 0.21 | 16796.88 | 1.0% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 14.56 | 0.16 | 14218.75 | 1.0% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 38.85 | 0.27 | 37939.45 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 15.55 | 0.17 | 15185.55 | 0.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 37.59 | 0.33 | 36708.98 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | faer | shared-session | not applicable | 16.27 | 0.20 | 15888.67 | 0.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 37.02 | 0.20 | 36152.34 | 0.6% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 2.24 | 0.01 | 2187.50 | 0.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 17.51 | 0.05 | 17099.61 | 0.5% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.78 | 0.00 | 2714.84 | 0.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 17.72 | 0.26 | 17304.69 | 1.1% | passed |
| solve_n4_count1024 | solve | n4 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | faer | shared-session | not applicable | 2.90 | 0.02 | 2832.03 | 1.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 18.10 | 0.21 | 17675.78 | 1.5% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 3.02 | 0.01 | 2949.22 | 0.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 18.24 | 0.10 | 17812.50 | 0.7% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 3.50 | 0.01 | 3417.97 | 0.2% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 18.66 | 0.08 | 18222.66 | 0.6% | passed |
| solve_n8_count1024 | solve | n8 | — | 8 | blas | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | faer | shared-session | not applicable | 3.72 | 0.02 | 3632.81 | 0.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 19.03 | 0.25 | 18583.98 | 0.9% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 207770.00 | 1030.62 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 14 | 55782.64 | 228.64 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 8 | 8 | 22 | 29610.59 | 273.23 | 1.1% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 3 | 163010.67 | 1716.50 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 20 | 44176.95 | 160.30 | 0.6% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 8 | 8 | 27 | 24619.15 | 253.81 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 153887.00 | 1115.10 | 0.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 20 | 41444.85 | 301.60 | 0.6% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 30 | 23324.77 | 224.93 | 1.6% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 12 | 73038.75 | 347.67 | 0.6% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 37 | 20508.00 | 82.58 | 0.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 52 | 12581.88 | 442.38 | 2.2% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 279885.67 | 1023.50 | 0.6% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 4 | 183424.75 | 1994.88 | 0.9% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 8 | 8 | 5 | 161118.00 | 695.30 | 0.7% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 280453.33 | 1706.50 | 0.7% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 5 | 184139.40 | 1045.10 | 0.3% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 8 | 8 | 5 | 160116.00 | 1369.60 | 0.5% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 57 | 11489.51 | 130.32 | 0.8% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 55 | 11826.25 | 151.65 | 10.7% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 8 | 8 | 54 | 12096.96 | 140.45 | 1.0% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 111 | 5996.05 | 47.79 | 2.0% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 98 | 7329.69 | 45.03 | 11.1% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 8 | 8 | 91 | 7533.58 | 72.61 | 0.8% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 12 | 68016.92 | 641.58 | 0.5% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 35 | 16027.34 | 208.67 | 1.2% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 8 | 8 | 33 | 10303.12 | 111.59 | 1.2% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 10 | 86421.20 | 612.15 | 0.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 33 | 23885.45 | 104.59 | 0.6% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 8 | 8 | 47 | 12772.28 | 274.99 | 2.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 155313.80 | 646.20 | 0.6% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 44501.75 | 1421.97 | 10.1% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 23 | 25889.09 | 2765.37 | 12.5% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 11 | 76527.36 | 454.41 | 0.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 36 | 20758.19 | 291.94 | 9.0% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 48 | 12538.15 | 277.80 | 2.2% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 20 | 32089.35 | 310.08 | 0.7% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 20 | 32394.95 | 446.60 | 10.3% | passed (NOISY) |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 8 | 8 | 20 | 32635.30 | 343.42 | 0.9% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 133 | 3742.70 | 37.93 | 2.0% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 126 | 4081.90 | 74.98 | 12.7% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 8 | 8 | 109 | 3999.92 | 79.92 | 1.7% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 45 | 13109.58 | 173.22 | 0.9% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 44 | 13380.66 | 203.67 | 1.8% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 8 | 8 | 38 | 13355.13 | 98.99 | 1.6% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.12 | 0.00 | 0.123 | 0.126 | 1.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.12 | 0.00 | 0.125 | 0.132 | 2.3% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.31 | 0.01 | 0.315 | 0.321 | 4.0% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.24 | 0.00 | 0.245 | 0.248 | 0.8% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.46 | 0.03 | 1.472 | 1.479 | 1.6% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 1.47 | 0.03 | 1.493 | 1.528 | 2.4% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 6.02 | 0.11 | 6.154 | 6.258 | 1.4% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 1.92 | 0.01 | 1.937 | 1.953 | 0.8% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.13 | 0.00 | 0.135 | 0.150 | 3.2% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.13 | 0.00 | 0.137 | 0.148 | 3.1% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.31 | 0.01 | 0.320 | 0.325 | 3.6% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.25 | 0.00 | 0.258 | 0.263 | 1.2% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.54 | 0.03 | 1.554 | 1.572 | 1.6% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.55 | 0.01 | 1.554 | 1.559 | 1.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 6.13 | 0.09 | 6.271 | 6.390 | 1.4% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 2.04 | 0.03 | 2.056 | 2.062 | 0.9% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 8 | 32 | 0.14 | 0.00 | 0.140 | 0.144 | 1.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 8 | 32 | 0.14 | 0.00 | 0.143 | 0.145 | 2.1% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 8 | 32 | 0.33 | 0.02 | 0.339 | 0.385 | 5.7% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 8 | 32 | 0.26 | 0.01 | 0.272 | 0.766 | 45.1% | passed (NOISY; inconclusive) |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 8 | 128 | 1.63 | 0.02 | 1.647 | 1.657 | 1.0% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 8 | 128 | 1.63 | 0.01 | 1.642 | 1.665 | 0.8% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 8 | 128 | 6.23 | 0.15 | 6.418 | 6.423 | 1.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 8 | 128 | 2.15 | 0.02 | 2.169 | 2.188 | 0.9% | passed |

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
  "latency_case_processes_seconds": 37.03669327800162,
  "separate_cache_diagnostics_seconds": 0.4680949589237571,
  "collection_total_seconds": 37.504788547055796
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 37.50060262205079,
  "separate_cache_diagnostics_seconds": 0.48206551698967814,
  "collection_total_seconds": 37.98266841005534
}
```

Wall observations `wall_collection_t8.json`:
```json
{
  "latency_case_processes_seconds": 39.42676964402199,
  "separate_cache_diagnostics_seconds": 0.5155168760102242,
  "collection_total_seconds": 39.94228675006889
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 130.844931952,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 0.144646039,
  "validation_metadata_and_collection_seconds": 130.621383924,
  "command_before_report_seconds": 130.766029963
}
```

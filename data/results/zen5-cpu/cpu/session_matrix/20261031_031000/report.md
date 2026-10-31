# CPU shared-session matrix results

Raw data and provenance: `data/results/zen5-cpu/cpu/session_matrix/20261031_031000`.

### run_t1

```yaml
timestamp: '2026-10-31T03:10:00Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: dae444a6a8f82e39fb37b05009ddc67d7d143d60
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-openblas BENCHMARK_TARGET_PROFILE=zen5-cpu
    /home/shinaoka/projects/tensor4all/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 8 '
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
timestamp: '2026-10-31T03:10:44Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: dae444a6a8f82e39fb37b05009ddc67d7d143d60
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-openblas BENCHMARK_TARGET_PROFILE=zen5-cpu
    /home/shinaoka/projects/tensor4all/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 8 '
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
timestamp: '2026-10-31T03:11:27Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-openblas
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark
  commit: dae444a6a8f82e39fb37b05009ddc67d7d143d60
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-openblas BENCHMARK_TARGET_PROFILE=zen5-cpu
    /home/shinaoka/projects/tensor4all/tenferro-benchmark/scripts/run_cpu_session.sh
    1 4 8 '
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
- Expected 171, selected 171, executed 72, unsupported 99, failed 0, missing 0, noisy 5.
- Complete: **NO** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Unsupported: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n8_count1024[faer]@t4`, `matmul_n16_count1024[faer]@t4`, `matmul_n32_count1024[faer]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4` (+59 more)
- Noisy: `matmul_n8_count1024[pytorch]@t1`, `solve_n4_count1024[blas]@t4`, `solve_n8_count1024[blas]@t4`, `matmul_n4_count1024[blas]@t8`, `matmul_n8_count1024[pytorch]@t8`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | blas | shared-session | not applicable | 1.97 | 0.01 | 1923.83 | 0.4% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.19 | 0.09 | 3115.23 | 6.1% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | blas | shared-session | not applicable | 2.60 | 0.01 | 2539.06 | 0.5% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.15 | 0.03 | 3076.17 | 4.8% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | blas | shared-session | not applicable | 2.75 | 0.03 | 2685.55 | 0.7% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 3.53 | 0.05 | 3447.27 | 5.2% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | blas | shared-session | not applicable | 1.60 | 0.01 | 1562.50 | 0.3% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 2.06 | 0.02 | 2011.72 | 6.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | blas | shared-session | not applicable | 2.20 | 0.01 | 2148.44 | 1.0% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 2.07 | 0.02 | 2021.48 | 6.2% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | blas | shared-session | not applicable | 2.27 | 0.02 | 2216.80 | 0.7% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 2.05 | 0.02 | 2001.95 | 6.7% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | blas | shared-session | not applicable | 4.74 | 0.08 | 4628.91 | 1.2% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 8.65 | 0.10 | 8447.27 | 2.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | blas | shared-session | not applicable | 5.33 | 0.04 | 5205.08 | 0.7% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 8.36 | 0.05 | 8164.06 | 2.3% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | blas | shared-session | not applicable | 5.54 | 0.06 | 5410.16 | 0.8% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 8.37 | 0.05 | 8173.83 | 2.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | blas | shared-session | not applicable | 1.64 | 0.01 | 1601.56 | 0.4% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.95 | 0.01 | 1904.30 | 6.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | blas | shared-session | not applicable | 2.27 | 0.02 | 2216.80 | 0.8% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.96 | 0.01 | 1914.06 | 6.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | blas | shared-session | not applicable | 2.31 | 0.02 | 2255.86 | 15.6% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 1.97 | 0.02 | 1923.83 | 7.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | blas | shared-session | not applicable | 1.69 | 0.01 | 1650.39 | 0.8% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.06 | 0.02 | 2011.72 | 12.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | blas | shared-session | not applicable | 2.27 | 0.01 | 2216.80 | 0.2% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.07 | 0.04 | 2021.48 | 6.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | blas | shared-session | not applicable | 2.39 | 0.01 | 2333.98 | 0.3% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 2.11 | 0.11 | 2060.55 | 10.1% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | blas | shared-session | not applicable | 8.09 | 0.49 | 7900.39 | 3.1% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 21.60 | 0.73 | 21093.75 | 2.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | blas | shared-session | not applicable | 7.20 | 0.46 | 7031.25 | 3.2% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 21.62 | 0.15 | 21113.28 | 1.4% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | blas | shared-session | not applicable | 7.87 | 0.56 | 7685.55 | 4.8% | passed |
| solve_n16_count1024 | solve | n16 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 22.36 | 0.16 | 21835.94 | 1.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | blas | shared-session | not applicable | 1.86 | 0.00 | 1816.41 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 17.30 | 0.08 | 16894.53 | 0.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | blas | shared-session | not applicable | 2.98 | 0.02 | 2910.16 | 0.6% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 17.13 | 0.13 | 16728.52 | 0.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | blas | shared-session | not applicable | 3.08 | 0.01 | 3007.81 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 17.41 | 0.04 | 17001.95 | 0.7% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | blas | shared-session | not applicable | 27.66 | 2.36 | 27011.72 | 4.3% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 38.65 | 0.18 | 37744.14 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | blas | shared-session | not applicable | 17.41 | 2.43 | 17001.95 | 6.8% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 36.12 | 0.16 | 35273.44 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | blas | shared-session | not applicable | 16.30 | 2.46 | 15917.97 | 7.3% | passed |
| solve_n32_count1024 | solve | n32 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 36.54 | 0.26 | 35683.59 | 1.0% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | blas | shared-session | not applicable | 2.23 | 0.01 | 2177.73 | 0.5% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 17.40 | 0.03 | 16992.19 | 0.8% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | blas | shared-session | not applicable | 3.64 | 0.01 | 3554.69 | 13.9% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 17.23 | 0.08 | 16826.17 | 0.7% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | blas | shared-session | not applicable | 3.62 | 0.01 | 3535.16 | 0.3% | passed |
| solve_n4_count1024 | solve | n4 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 17.35 | 0.14 | 16943.36 | 1.2% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | blas | shared-session | not applicable | 3.34 | 0.01 | 3261.72 | 0.5% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 18.35 | 0.08 | 17919.92 | 1.3% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | blas | shared-session | not applicable | 4.36 | 0.02 | 4257.81 | 12.1% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 19.17 | 0.07 | 18720.70 | 0.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | blas | shared-session | not applicable | 4.66 | 0.01 | 4550.78 | 6.5% | passed |
| solve_n8_count1024 | solve | n8 | — | 8 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 18.14 | 0.04 | 17714.84 | 1.0% | passed |

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
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | 128 | 1.43 | 0.03 | 1.458 | 1.495 | 2.0% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | 128 | 1.44 | 0.01 | 1.448 | 1.450 | 0.9% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | 128 | 5.82 | 0.11 | 5.998 | 6.202 | 2.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | 128 | 2.16 | 0.02 | 2.179 | 2.183 | 0.7% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | 128 | 1.58 | 0.02 | 1.591 | 1.594 | 0.9% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | 128 | 1.60 | 0.02 | 1.618 | 1.622 | 1.9% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | 128 | 6.06 | 0.17 | 6.228 | 6.279 | 1.9% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | 128 | 2.32 | 0.01 | 2.334 | 2.347 | 0.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 8 | 128 | 1.61 | 0.01 | 1.617 | 1.624 | 0.6% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 8 | 128 | 1.64 | 0.01 | 1.652 | 1.655 | 0.4% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 8 | 128 | 5.98 | 0.14 | 6.185 | 6.311 | 2.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 8 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 8 | 128 | 2.40 | 0.01 | 2.413 | 2.418 | 0.4% | passed |
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
  "latency_case_processes_seconds": 37.487708812986966,
  "separate_cache_diagnostics_seconds": 0.5472508610109799,
  "collection_total_seconds": 38.034960174991284
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 37.825727079005446,
  "separate_cache_diagnostics_seconds": 0.6129310479736887,
  "collection_total_seconds": 38.43865850800648
}
```

Wall observations `wall_collection_t8.json`:
```json
{
  "latency_case_processes_seconds": 39.75909317401238,
  "separate_cache_diagnostics_seconds": 0.6945944670005701,
  "collection_total_seconds": 40.45368811202934
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 133.548791051,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 0.146266855,
  "validation_metadata_and_collection_seconds": 133.324152771,
  "command_before_report_seconds": 133.470419626
}
```

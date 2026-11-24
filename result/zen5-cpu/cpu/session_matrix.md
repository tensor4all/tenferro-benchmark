# CPU shared-session matrix results

Raw data and provenance: `data/results/zen5-cpu/cpu/session_matrix/20261124_053537`.

### run_t1

```yaml
timestamp: '2026-11-24T05:35:37Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/work/zen5-refresh-20261124/harness/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/zen5-refresh-20261124/harness
  commit: bb9869533cc2ad586b71cdb49a3e1f85439cd2b2
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=zen5-cpu /home/shinaoka/work/zen5-refresh-20261124/harness/scripts/run_cpu_session.sh
    1 4 8 '
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
timestamp: '2026-11-24T05:36:22Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/work/zen5-refresh-20261124/harness/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/zen5-refresh-20261124/harness
  commit: bb9869533cc2ad586b71cdb49a3e1f85439cd2b2
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=zen5-cpu /home/shinaoka/work/zen5-refresh-20261124/harness/scripts/run_cpu_session.sh
    1 4 8 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 5
  selection_filter: null
  expected_cases: 33
  selected_cases: 33
```

### run_t8

```yaml
timestamp: '2026-11-24T05:37:05Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/work/zen5-refresh-20261124/harness/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/zen5-refresh-20261124/harness
  commit: bb9869533cc2ad586b71cdb49a3e1f85439cd2b2
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 BENCHMARK_TARGET_PROFILE=zen5-cpu /home/shinaoka/work/zen5-refresh-20261124/harness/scripts/run_cpu_session.sh
    1 4 8 '
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
- Expected 129, selected 129, executed 129, unsupported 0, failed 0, skipped 0, missing 0, noisy 6.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 129 | 129 | 129 | 0 | 0 | 0 | 0 | 6 |

- Noisy: `matmul_n2_count1024[pytorch]@t1`, `matmul_n32_count1024[pytorch]@t1`, `matmul_n4_count1024[pytorch]@t4`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t4`, `matmul_n4_count1024[pytorch]@t8`, `hadamard_f64_m64n64_dot_alloc[faer]@t8`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | not applicable | 2.34 | 0.03 | 2285.16 | 1.1% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.41 | 0.22 | 3330.08 | 7.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | not applicable | 2.72 | 0.03 | 2656.25 | 1.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.30 | 0.17 | 3222.66 | 6.6% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | faer | shared-session | not applicable | 3.07 | 0.08 | 2998.05 | 4.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 3.52 | 0.17 | 3437.50 | 5.8% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | not applicable | 2.04 | 0.01 | 1992.19 | 1.9% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.97 | 0.42 | 1923.83 | 12.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.39 | 0.01 | 2333.98 | 0.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.98 | 0.03 | 1933.59 | 8.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | faer | shared-session | not applicable | 2.49 | 0.01 | 2431.64 | 0.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 2.14 | 0.26 | 2089.84 | 9.5% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | not applicable | 5.40 | 0.07 | 5273.44 | 1.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 8.75 | 0.24 | 8544.92 | 12.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | not applicable | 5.87 | 0.07 | 5732.42 | 1.1% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 8.46 | 0.09 | 8261.72 | 4.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | faer | shared-session | not applicable | 6.24 | 0.08 | 6093.75 | 0.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 8.29 | 0.17 | 8095.70 | 1.9% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | not applicable | 2.05 | 0.01 | 2001.95 | 3.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.89 | 0.06 | 1845.70 | 6.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.48 | 0.04 | 2421.88 | 1.1% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.86 | 0.02 | 1816.41 | 10.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | faer | shared-session | not applicable | 2.48 | 0.01 | 2421.88 | 0.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 1.94 | 0.23 | 1894.53 | 11.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | not applicable | 2.09 | 0.01 | 2041.02 | 0.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.05 | 0.08 | 2001.95 | 8.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | not applicable | 2.50 | 0.02 | 2441.41 | 0.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.13 | 0.25 | 2080.08 | 8.3% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | faer | shared-session | not applicable | 2.53 | 0.01 | 2470.70 | 0.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 2.04 | 0.03 | 1992.19 | 8.3% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | not applicable | 5.44 | 0.14 | 5312.50 | 1.7% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 21.35 | 0.34 | 20849.61 | 1.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | not applicable | 5.88 | 0.12 | 5742.19 | 2.2% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 22.08 | 0.61 | 21562.50 | 2.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | faer | shared-session | not applicable | 6.38 | 0.09 | 6230.47 | 1.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 8 | pytorch | shared-session | not applicable | 22.25 | 0.47 | 21728.52 | 1.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | not applicable | 2.12 | 0.01 | 2070.31 | 0.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 17.45 | 0.21 | 17041.02 | 0.9% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | not applicable | 2.58 | 0.05 | 2519.53 | 1.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 17.02 | 0.23 | 16621.09 | 1.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | faer | shared-session | not applicable | 2.74 | 0.01 | 2675.78 | 0.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 8 | pytorch | shared-session | not applicable | 16.98 | 0.10 | 16582.03 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | not applicable | 14.75 | 0.28 | 14404.30 | 1.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 39.35 | 0.48 | 38427.73 | 1.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | not applicable | 15.82 | 0.25 | 15449.22 | 1.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 36.91 | 0.43 | 36044.92 | 2.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | faer | shared-session | not applicable | 16.29 | 0.25 | 15908.20 | 1.2% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 8 | pytorch | shared-session | not applicable | 37.02 | 0.53 | 36152.34 | 2.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | not applicable | 2.25 | 0.00 | 2197.27 | 0.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 16.86 | 0.16 | 16464.84 | 1.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | not applicable | 2.71 | 0.01 | 2646.48 | 0.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 17.35 | 0.23 | 16943.36 | 1.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | faer | shared-session | not applicable | 3.00 | 0.01 | 2929.69 | 0.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 8 | pytorch | shared-session | not applicable | 17.85 | 0.64 | 17431.64 | 2.2% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | not applicable | 3.07 | 0.01 | 2998.05 | 0.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 18.07 | 0.14 | 17646.48 | 1.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | not applicable | 3.54 | 0.01 | 3457.03 | 0.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 18.05 | 0.19 | 17626.95 | 1.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | faer | shared-session | not applicable | 3.79 | 0.01 | 3701.17 | 0.2% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 8 | pytorch | shared-session | not applicable | 18.17 | 0.14 | 17744.14 | 0.9% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples. tenferro-rs #2004 removed the batch-policy API, so cases whose manifest policy is an override report `unsupported` instead of being measured with the compiled backend default, and the batched and stream cases are declared for the faer provider, so a build without it reports them `unsupported` too. Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest, never from timing. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 4 | 208068.75 | 1907.25 | 1.9% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 55959.50 | 1085.75 | 9.8% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 8 | 8 | 25 | 29674.36 | 424.18 | 1.0% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 161884.60 | 752.30 | 1.3% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 14 | 61299.50 | 605.43 | 1.0% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 8 | 8 | 32 | 24046.88 | 354.55 | 1.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 6 | 152965.33 | 822.33 | 0.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 15 | 41344.27 | 375.03 | 7.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 28 | 23157.71 | 186.59 | 0.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 12 | 69900.58 | 403.62 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 38 | 19511.76 | 114.03 | 1.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 50 | 12412.70 | 1673.20 | 8.2% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 279729.33 | 1826.50 | 0.5% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 5 | 183584.80 | 3816.90 | 1.6% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 8 | 8 | 5 | 159095.40 | 817.50 | 0.7% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 3 | 277505.00 | 1402.50 | 0.4% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 5 | 180160.40 | 2323.30 | 1.3% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 8 | 8 | 5 | 160255.60 | 3601.70 | 1.4% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 60 | 11550.30 | 75.05 | 0.5% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 57 | 11759.25 | 159.51 | 12.4% | passed (NOISY) |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 8 | 8 | 54 | 12052.61 | 60.21 | 0.7% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 105 | 6293.61 | 46.19 | 2.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 83 | 6807.13 | 153.78 | 2.1% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 8 | 8 | 99 | 6734.75 | 48.63 | 0.7% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 14 | 67820.14 | 685.86 | 1.6% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 35 | 16070.51 | 190.37 | 1.4% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 8 | 8 | 52 | 10404.21 | 35.25 | 0.7% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 10 | 87181.80 | 852.50 | 1.9% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 34 | 24288.91 | 560.16 | 1.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 8 | 8 | 50 | 14694.90 | 42.08 | 0.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 155052.00 | 1080.10 | 0.5% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 43897.25 | 644.62 | 9.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 23 | 26010.17 | 185.52 | 0.9% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 12 | 71107.83 | 424.46 | 0.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 33 | 21119.88 | 375.38 | 2.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 8 | 8 | 35 | 12875.20 | 177.47 | 2.0% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 21 | 34594.29 | 389.76 | 1.3% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 20 | 35246.00 | 439.30 | 8.9% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 8 | 8 | 19 | 35668.95 | 299.50 | 1.0% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 126 | 3601.01 | 57.37 | 1.2% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 126 | 3981.40 | 65.68 | 1.9% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 8 | 8 | 123 | 3959.33 | 40.03 | 22.9% | passed (NOISY) |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 45 | 14390.53 | 150.26 | 1.1% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 43 | 14745.12 | 320.24 | 1.4% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 8 | 8 | 43 | 14662.19 | 135.49 | 1.0% | passed |

## Whole shape-sequence public-call diagnostics

One sample is the entire ordered sequence, with required planning/cache lookup inside the clock. Inputs, views, wrapping, bindings and retention containers are prepared first; outputs survive clock stop. The initialization pass and warmups are separate. Exact keys and ordering for every pass are in the raw row. Cold mode clears only the declared eager extension caches outside each sequence; it does not bypass lookup or retention. Capacity thrash bounds that store to one entry with four revisited keys. Concrete and eager three-operand streams are distinct workloads.

| Case | Provider | Threads | Calls/sequence | Median sequence ms | IQR ms | p90 ms | Maximum ms | CoV | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stream_f64_fixed_len32_einsum_alloc | faer | 1 | 32 | 0.13 | 0.00 | 0.136 | 0.142 | 3.9% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 1 | 32 | 0.13 | 0.00 | 0.130 | 0.131 | 0.9% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 1 | 32 | 0.31 | 0.01 | 0.321 | 0.328 | 4.1% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 1 | 32 | 0.25 | 0.00 | 0.252 | 0.259 | 1.6% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | 128 | 1.46 | 0.05 | 1.509 | 1.545 | 2.6% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | 128 | 1.50 | 0.03 | 1.562 | 1.756 | 4.6% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | 128 | 6.04 | 0.10 | 6.317 | 6.498 | 2.5% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | 128 | 2.02 | 0.01 | 2.040 | 2.046 | 0.7% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | 32 | 0.14 | 0.01 | 0.161 | 0.170 | 7.2% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | 32 | 0.14 | 0.00 | 0.148 | 0.157 | 2.6% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | 32 | 0.33 | 0.02 | 0.344 | 0.348 | 4.5% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | 32 | 0.26 | 0.00 | 0.268 | 0.270 | 1.2% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | 128 | 1.59 | 0.04 | 1.625 | 1.689 | 2.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | 128 | 1.60 | 0.01 | 1.626 | 1.635 | 1.1% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | 128 | 6.26 | 0.24 | 6.527 | 6.560 | 2.6% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 4 | 128 | 2.07 | 0.02 | 2.157 | 2.316 | 3.4% | passed |
| stream_f64_fixed_len32_einsum_alloc | faer | 8 | 32 | 0.15 | 0.00 | 0.147 | 0.151 | 1.3% | passed |
| stream_f64_mixed_len32_einsum_alloc | faer | 8 | 32 | 0.15 | 0.00 | 0.149 | 0.153 | 1.5% | passed |
| stream_f64_fresh_len32_einsum_alloc | faer | 8 | 32 | 0.34 | 0.01 | 0.343 | 0.364 | 3.7% | passed |
| stream_f64_strides_len32_einsum_alloc | faer | 8 | 32 | 0.27 | 0.00 | 0.275 | 0.276 | 0.6% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 8 | 128 | 1.60 | 0.03 | 1.635 | 1.637 | 1.5% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 8 | 128 | 1.64 | 0.02 | 1.655 | 1.667 | 0.7% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 8 | 128 | 6.26 | 0.12 | 6.420 | 6.486 | 1.4% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 8 | 128 | 2.15 | 0.01 | 2.151 | 2.157 | 0.3% | passed |

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
  "latency_case_processes_seconds": 39.57793913700152,
  "separate_cache_diagnostics_seconds": 0.4766493440256454,
  "collection_total_seconds": 40.054588821018115
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 37.655703254044056,
  "separate_cache_diagnostics_seconds": 0.4976418449659832,
  "collection_total_seconds": 38.153345339989755
}
```

Wall observations `wall_collection_t8.json`:
```json
{
  "latency_case_processes_seconds": 39.08386611397145,
  "separate_cache_diagnostics_seconds": 0.5089634780306369,
  "collection_total_seconds": 39.592829842003994
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 133.552223339,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 0.149220002,
  "validation_metadata_and_collection_seconds": 133.325547079,
  "command_before_report_seconds": 133.474767081
}
```

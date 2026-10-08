# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261008_062752`.

### run_t1

```yaml
timestamp: '2026-10-08T06:27:52Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
  dirty: false
collection:
  command: 'BENCH_EFFORT=standard BENCH_COVERAGE=full TENFERRO_CPU_FEATURES=system-mkl
    BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_cpu_session.sh 1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: full
  manifest_version: 3
  selection_filter: null
  expected_cases: 68
  selected_cases: 68
```

### run_t4

```yaml
timestamp: '2026-10-08T06:28:28Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 2cd0fd6445dcd9aad98990f76c17e33e773c4a7d
  dirty: false
collection:
  command: 'BENCH_EFFORT=standard BENCH_COVERAGE=full TENFERRO_CPU_FEATURES=system-mkl
    BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_cpu_session.sh 1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: full
  manifest_version: 3
  selection_filter: null
  expected_cases: 68
  selected_cases: 68
```

## Case status

- Manifest: `cpu/session_matrix` version 3, coverage `full`, effort `standard`
- Expected 176, selected 176, executed 170, unsupported 6, failed 0, missing 0, noisy 22.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Unsupported: `bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor[faer]@t4`, `bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor[faer]@t4`
- Noisy: `matmul_n2_count1024[blas]@t1`, `matmul_n2_count1024[pytorch]@t1`, `matmul_n4_count1024[blas]@t1`, `matmul_n4_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b320_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b4096_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n32_count1024[blas]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_strided_alloc_auto[faer]@t4`, `bdot_c64_b1024_m4n4k4_strided_into_auto[faer]@t4`, `bdot_f64_b1024_m4n4k4_direct_into_forced-sequential[faer]@t4`, `hadamard_f64_m256n256_einsum_alloc[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 2.38 | 0.01 | 2324.22 | 8.0% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | Managed | 1.73 | 0.01 | 1689.45 | 0.4% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 2.42 | 0.03 | 2363.28 | 6.0% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 2.39 | 0.02 | 2333.98 | 7.1% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | Managed | 1.78 | 0.00 | 1738.28 | 0.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 2.46 | 0.02 | 2402.34 | 5.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.04 | 0.00 | 1015.62 | 14.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | Managed | 0.82 | 0.01 | 800.78 | 1.1% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.57 | 0.17 | 1533.20 | 10.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.04 | 0.01 | 1015.62 | 3.6% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | Managed | 0.83 | 0.01 | 810.55 | 10.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.65 | 0.02 | 1611.33 | 7.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 7.43 | 0.09 | 7255.86 | 0.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | Managed | 4.57 | 0.09 | 4462.89 | 1.1% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 6.31 | 0.04 | 6162.11 | 3.2% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 7.78 | 1.94 | 7597.66 | 17.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | Managed | 4.63 | 0.12 | 4521.48 | 4.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 6.56 | 0.16 | 6406.25 | 9.3% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.10 | 0.01 | 1074.22 | 12.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | Managed | 0.88 | 0.02 | 859.38 | 14.6% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.55 | 0.02 | 1513.67 | 6.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.10 | 0.01 | 1074.22 | 8.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | Managed | 0.89 | 0.00 | 869.14 | 13.8% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.55 | 0.02 | 1513.67 | 7.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.31 | 0.01 | 1279.30 | 3.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | Managed | 1.01 | 0.01 | 986.33 | 0.5% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 1.65 | 0.01 | 1611.33 | 6.0% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.31 | 0.01 | 1279.30 | 0.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | Managed | 1.01 | 0.01 | 986.33 | 3.1% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 1.67 | 0.02 | 1630.86 | 7.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 4.77 | 0.01 | 4658.20 | 0.3% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | Managed | 4.70 | 0.02 | 4589.84 | 0.3% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 17.01 | 0.04 | 16611.33 | 0.6% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 4.79 | 0.02 | 4677.73 | 3.0% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | Managed | 4.73 | 0.03 | 4619.14 | 0.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 17.50 | 0.08 | 17089.84 | 1.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.22 | 0.01 | 1191.41 | 0.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | Managed | 1.39 | 0.01 | 1357.42 | 0.5% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 13.63 | 0.06 | 13310.55 | 0.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.22 | 0.00 | 1191.41 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | Managed | 1.39 | 0.00 | 1357.42 | 0.3% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 13.42 | 0.10 | 13105.47 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 18.60 | 0.12 | 18164.06 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | Managed | 12.76 | 0.04 | 12460.94 | 0.4% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 30.42 | 0.15 | 29707.03 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 17.65 | 0.08 | 17236.33 | 0.4% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | Managed | 12.67 | 0.08 | 12373.05 | 0.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 28.51 | 0.14 | 27841.80 | 0.6% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.31 | 0.01 | 1279.30 | 0.6% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | Managed | 1.57 | 0.02 | 1533.20 | 1.6% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 13.45 | 0.07 | 13134.77 | 0.9% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.31 | 0.01 | 1279.30 | 0.7% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | Managed | 1.60 | 0.01 | 1562.50 | 0.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 13.58 | 0.15 | 13261.72 | 1.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 2.33 | 0.01 | 2275.39 | 0.4% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | Managed | 2.35 | 0.02 | 2294.92 | 9.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 14.34 | 0.07 | 14003.91 | 0.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 2.34 | 0.02 | 2285.16 | 2.1% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | Managed | 2.36 | 0.01 | 2304.69 | 6.3% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 14.48 | 0.09 | 14140.62 | 1.1% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples; policy overrides applied before session entry (backend) or around all samples (scoped). Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest; provider-call/lane mechanics are in the separate `cpu/route_contract` diagnostics, never in these timings. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_c64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | c64 | 1024×4×4×4 | 1 | 1 | 11 | 131059.55 | 1166.73 | 2.7% | passed |
| bdot_c64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | c64 | 1024×4×4×4 | 4 | 4 | 14 | 52616.21 | 1865.29 | 3.3% | passed |
| bdot_c64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | c64 | 1024×4×4×4 | 1 | 1 | 5 | 85320.60 | 574.00 | 0.5% | passed |
| bdot_c64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | c64 | 1024×4×4×4 | 4 | 4 | 16 | 27284.50 | 298.66 | 1.7% | passed |
| bdot_c64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | c64 | 1024×4×4×4 | 1 | 1 | 7 | 127873.71 | 642.00 | 0.5% | passed |
| bdot_c64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | c64 | 1024×4×4×4 | 4 | 4 | 16 | 53454.38 | 2862.22 | 3.7% | passed |
| bdot_c64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | c64 | 1024×4×4×4 | 1 | 1 | 10 | 87344.50 | 263.50 | 0.2% | passed |
| bdot_c64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | c64 | 1024×4×4×4 | 4 | 4 | 12 | 27172.00 | 700.88 | 39.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 6 | 157282.83 | 1825.08 | 2.7% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 52876.44 | 3983.75 | 11.3% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 7 | 135164.43 | 672.64 | 0.3% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 8 | 40662.88 | 1043.19 | 5.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 7 | 83800.57 | 2036.71 | 27.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 35543.19 | 1553.88 | 7.8% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_backend-outer-min-items-4096 | concrete / allocating / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | 1 | 13 | 85746.08 | 128.69 | 0.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_backend-outer-min-items-4096 | concrete / allocating / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | 4 | 14 | 84981.50 | 434.71 | 0.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel | concrete / allocating / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 1 | 1 | — | — | — | — | unsupported: dot_general: unsupported operation: batch strategy OuterParallel is not available: the selected CPU  |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-outer-parallel | concrete / allocating / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 4 | 4 | 13 | 34407.15 | 1024.19 | 2.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-provider-items | concrete / allocating / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 1 | 1 | 14 | 85023.00 | 601.50 | 1.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-provider-items | concrete / allocating / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 4 | 4 | 14 | 85445.21 | 538.50 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-sequential | concrete / allocating / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 1 | 1 | 14 | 83943.79 | 315.57 | 0.6% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-sequential | concrete / allocating / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 4 | 4 | 14 | 84821.86 | 593.21 | 1.0% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor | concrete / allocating / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 1 | 1 | — | — | — | — | unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavai |
| bdot_f64_b1024_m4n4k4_direct_alloc_forced-whole-batch-vendor | concrete / allocating / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 4 | 4 | — | — | — | — | unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavai |
| bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096 | concrete / allocating / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | 1 | 14 | 84732.43 | 451.21 | 13.6% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_alloc_scoped-outer-min-items-4096 | concrete / allocating / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | 4 | 10 | 85348.80 | 1262.95 | 1.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 7 | 62461.86 | 483.71 | 0.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 19 | 22717.00 | 710.79 | 2.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096 | concrete / reused / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | 1 | 14 | 63241.21 | 200.39 | 13.8% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_backend-outer-min-items-4096 | concrete / reused / owned / direct | auto+backend(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | 4 | 14 | 63165.36 | 229.75 | 1.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel | concrete / reused / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 1 | 1 | — | — | — | — | unsupported: dot_general: unsupported operation: batch strategy OuterParallel is not available: the selected CPU  |
| bdot_f64_b1024_m4n4k4_direct_into_forced-outer-parallel | concrete / reused / owned / direct | forced:OuterParallel | f64 | 1024×4×4×4 | 4 | 4 | 17 | 22728.65 | 744.65 | 3.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_forced-provider-items | concrete / reused / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 1 | 1 | 14 | 63239.79 | 80.50 | 0.2% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_forced-provider-items | concrete / reused / owned / direct | forced:ProviderItems | f64 | 1024×4×4×4 | 4 | 4 | 14 | 63145.36 | 173.14 | 3.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_forced-sequential | concrete / reused / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 1 | 1 | 14 | 62891.29 | 128.46 | 0.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_forced-sequential | concrete / reused / owned / direct | forced:Sequential | f64 | 1024×4×4×4 | 4 | 4 | 14 | 63262.00 | 774.71 | 11.5% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor | concrete / reused / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 1 | 1 | — | — | — | — | unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavai |
| bdot_f64_b1024_m4n4k4_direct_into_forced-whole-batch-vendor | concrete / reused / owned / direct | forced:WholeBatchVendor | f64 | 1024×4×4×4 | 4 | 4 | — | — | — | — | unsupported: dot_general: unsupported operation: configured CPU GEMM provider reported unsupported: RuntimeUnavai |
| bdot_f64_b1024_m4n4k4_direct_into_scoped-outer-min-items-4096 | concrete / reused / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 1 | 1 | 7 | 62071.14 | 438.71 | 0.5% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_scoped-outer-min-items-4096 | concrete / reused / owned / direct | auto+scoped(outer_min_items=4096) | f64 | 1024×4×4×4 | 4 | 4 | 15 | 62451.60 | 154.93 | 0.9% | passed |
| bdot_f64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 85707.14 | 149.21 | 0.3% | passed |
| bdot_f64_b1024_m4n4k4_strided_alloc_auto | concrete / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 21 | 34813.62 | 1936.05 | 13.8% | passed (NOISY) |
| bdot_f64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 15 | 59703.13 | 227.73 | 0.3% | passed |
| bdot_f64_b1024_m4n4k4_strided_into_auto | concrete / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 16 | 22441.00 | 824.72 | 3.7% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 6 | 206350.33 | 2306.83 | 1.3% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 10 | 90900.20 | 4326.60 | 3.3% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 6 | 127568.67 | 2445.50 | 13.7% | passed (NOISY) |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 8 | 37528.12 | 437.00 | 1.9% | passed |
| bdot_f64_b256_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 256×4×4×4 | 1 | 1 | 51 | 22477.02 | 237.59 | 5.7% | passed |
| bdot_f64_b256_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 256×4×4×4 | 4 | 4 | 52 | 22699.08 | 283.53 | 1.7% | passed |
| bdot_f64_b256_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 256×4×4×4 | 1 | 1 | 53 | 16799.94 | 29.68 | 0.4% | passed |
| bdot_f64_b256_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 256×4×4×4 | 4 | 4 | 52 | 16797.21 | 111.74 | 0.4% | passed |
| bdot_f64_b320_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 320×4×4×4 | 1 | 1 | 30 | 27677.43 | 360.67 | 10.1% | passed (NOISY) |
| bdot_f64_b320_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 320×4×4×4 | 4 | 4 | 21 | 21006.24 | 1233.50 | 6.0% | passed |
| bdot_f64_b320_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 320×4×4×4 | 1 | 1 | 43 | 20734.16 | 50.56 | 0.2% | passed |
| bdot_f64_b320_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 320×4×4×4 | 4 | 4 | 43 | 16969.16 | 913.35 | 5.2% | passed |
| bdot_f64_b4096_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 4096×4×4×4 | 1 | 1 | 3 | 248317.67 | 15223.83 | 8.3% | passed |
| bdot_f64_b4096_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 4096×4×4×4 | 4 | 4 | 4 | 75314.50 | 6027.62 | 7.8% | passed |
| bdot_f64_b4096_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 4096×4×4×4 | 1 | 1 | 3 | 254235.33 | 2524.67 | 11.6% | passed (NOISY) |
| bdot_f64_b4096_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 4096×4×4×4 | 4 | 4 | 5 | 76191.40 | 6641.60 | 7.0% | passed |
| bdot_f64_b64_m16n16k16_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×16×16×16 | 1 | 1 | 38 | 44184.26 | 93.20 | 0.3% | passed |
| bdot_f64_b64_m16n16k16_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×16×16×16 | 4 | 4 | 17 | 36666.12 | 2193.29 | 6.2% | passed |
| bdot_f64_b64_m16n16k16_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×16×16×16 | 1 | 1 | 40 | 21135.80 | 27.55 | 4.6% | passed |
| bdot_f64_b64_m16n16k16_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×16×16×16 | 4 | 4 | 28 | 16531.86 | 810.46 | 2.9% | passed |
| bdot_f64_b64_m32n32k32_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×32×32×32 | 1 | 1 | 10 | 155569.90 | 1156.15 | 0.7% | passed |
| bdot_f64_b64_m32n32k32_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×32×32×32 | 4 | 4 | 6 | 80366.33 | 3834.83 | 3.9% | passed |
| bdot_f64_b64_m32n32k32_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×32×32×32 | 1 | 1 | 10 | 75334.90 | 617.20 | 0.7% | passed |
| bdot_f64_b64_m32n32k32_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×32×32×32 | 4 | 4 | 14 | 26371.86 | 429.00 | 2.7% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 122 | 6946.85 | 43.40 | 0.4% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 124 | 7008.03 | 33.17 | 1.3% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_backend-lane-min-work-0 | concrete / allocating / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | 1 | 124 | 6981.21 | 59.95 | 0.5% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_backend-lane-min-work-0 | concrete / allocating / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | 4 | 76 | 8830.32 | 161.16 | 3.9% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_scoped-lane-min-work-0 | concrete / allocating / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | 1 | 129 | 7002.64 | 21.40 | 0.4% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_scoped-lane-min-work-0 | concrete / allocating / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | 4 | 79 | 8796.81 | 110.34 | 3.2% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 74 | 5176.50 | 35.74 | 0.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 145 | 5238.69 | 14.48 | 9.4% | passed |
| bdot_f64_b64_m4n4k4_direct_into_backend-lane-min-work-0 | concrete / reused / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | 1 | 72 | 5211.33 | 38.34 | 0.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_backend-lane-min-work-0 | concrete / reused / owned / direct | auto+backend(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | 4 | 49 | 7628.45 | 120.12 | 2.5% | passed |
| bdot_f64_b64_m4n4k4_direct_into_scoped-lane-min-work-0 | concrete / reused / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 1 | 1 | 142 | 5200.01 | 36.40 | 0.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_scoped-lane-min-work-0 | concrete / reused / owned / direct | auto+scoped(lane_min_work_ns=0) | f64 | 64×4×4×4 | 4 | 4 | 64 | 7791.23 | 85.71 | 2.5% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 15 | 60405.73 | 155.63 | 0.2% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 24 | 34560.08 | 730.96 | 3.8% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 13 | 54243.69 | 323.69 | 0.4% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 20 | 36883.45 | 708.85 | 8.8% | passed |
| bdot_f64_b64_m8n8k8_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×8×8×8 | 1 | 1 | 105 | 12861.08 | 161.11 | 3.6% | passed |
| bdot_f64_b64_m8n8k8_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×8×8×8 | 4 | 4 | 107 | 12709.07 | 148.36 | 3.6% | passed |
| bdot_f64_b64_m8n8k8_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×8×8×8 | 1 | 1 | 114 | 6641.62 | 17.63 | 0.5% | passed |
| bdot_f64_b64_m8n8k8_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×8×8×8 | 4 | 4 | 58 | 6768.43 | 67.36 | 2.2% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 10 | 94508.90 | 366.70 | 3.1% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 14 | 42766.21 | 5160.43 | 20.6% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 62806.86 | 263.32 | 1.4% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 14 | 23313.93 | 601.50 | 1.9% | passed |
| beinsum_f64_b1024_m4n4k4_strided_alloc_auto | concrete-einsum / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 7 | 94674.00 | 680.50 | 0.5% | passed |
| beinsum_f64_b1024_m4n4k4_strided_alloc_auto | concrete-einsum / allocating / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 13 | 50212.23 | 1652.77 | 4.6% | passed |
| beinsum_f64_b1024_m4n4k4_strided_into_auto | concrete-einsum / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 1 | 1 | 14 | 62666.57 | 677.71 | 2.6% | passed |
| beinsum_f64_b1024_m4n4k4_strided_into_auto | concrete-einsum / reused / view / direct-strided | auto | f64 | 1024×4×4×4 | 4 | 4 | 14 | 24736.57 | 505.25 | 2.0% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 25 | 22481.16 | 177.54 | 0.9% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 11 | 22638.09 | 1416.36 | 11.2% | passed (NOISY) |
| chain3_f64_n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×64 | 1 | 1 | 11 | 46164.18 | 486.86 | 0.9% | passed |
| chain3_f64_n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×64 | 4 | 4 | 13 | 45585.77 | 475.15 | 1.7% | passed |
| hadamard_f64_m256n256_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×256×256×1 | 1 | 1 | 25 | 134990.20 | 6177.24 | 8.9% | passed |
| hadamard_f64_m256n256_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×256×256×1 | 4 | 4 | 12 | 146470.75 | 384.92 | 0.3% | passed |
| hadamard_f64_m256n256_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×256×256×1 | 1 | 1 | 21 | 108558.71 | 3652.12 | 9.8% | passed |
| hadamard_f64_m256n256_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×256×256×1 | 4 | 4 | 5 | 108434.20 | 23953.20 | 11.7% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 77 | 10041.51 | 265.64 | 9.9% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 137 | 10116.64 | 60.55 | 0.8% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 57 | 15030.23 | 155.04 | 0.8% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 57 | 15062.05 | 148.70 | 9.6% | passed |
| stream_f64_fixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9532.62 | 113.17 | 0.7% | passed |
| stream_f64_fixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9517.94 | 95.02 | 2.9% | passed |
| stream_f64_fresh_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9595.56 | 227.62 | 34.7% | passed (NOISY) |
| stream_f64_fresh_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9503.84 | 203.81 | 2.4% | passed |
| stream_f64_mixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9538.56 | 166.39 | 34.2% | passed (NOISY) |
| stream_f64_mixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9409.28 | 93.12 | 1.2% | passed |
| stream_f64_strides_len32_einsum_alloc | concrete-einsum / allocating / view / direct-strided | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9517.59 | 115.20 | 1.0% | passed |
| stream_f64_strides_len32_einsum_alloc | concrete-einsum / allocating / view / direct-strided | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9509.47 | 105.05 | 1.2% | passed |

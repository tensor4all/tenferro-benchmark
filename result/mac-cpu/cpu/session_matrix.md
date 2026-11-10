# CPU shared-session matrix results

Raw data and provenance: `data/results/mac-cpu/cpu/session_matrix/20261110_175925`.

### run_t1

```yaml
timestamp: '2026-11-10T17:59:26Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 15cf8f674203aeca079999fdb339cf4d35801b88
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
timestamp: '2026-11-10T17:59:49Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark
  commit: 15cf8f674203aeca079999fdb339cf4d35801b88
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
- Expected 114, selected 114, executed 48, unsupported 66, failed 0, skipped 0, missing 0, noisy 8.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 114 | 48 | 66 | 0 | 0 | 0 | 8 |

- Unsupported: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n8_count1024[faer]@t4`, `matmul_n16_count1024[faer]@t4`, `matmul_n32_count1024[faer]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4` (+26 more)
- Noisy: `matmul_n16_count1024[pytorch]@t1`, `matmul_n32_count1024[pytorch]@t1`, `matmul_n2_count1024[blas]@t4`, `matmul_n8_count1024[blas]@t4`, `matmul_n16_count1024[pytorch]@t4`, `solve_n2_count1024[blas]@t4`, `solve_n4_count1024[blas]@t4`, `solve_n8_count1024[blas]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | accelerate | shared-session | not applicable | 1.19 | 0.02 | 1162.11 | 0.8% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 1.48 | 0.15 | 1445.31 | 13.8% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | accelerate | shared-session | not applicable | 1.87 | 0.07 | 1826.17 | 4.1% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 1.44 | 0.29 | 1406.25 | 11.5% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | accelerate | shared-session | not applicable | 0.80 | 0.08 | 781.25 | 8.2% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 0.86 | 0.07 | 839.84 | 6.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | accelerate | shared-session | not applicable | 1.31 | 0.18 | 1279.30 | 38.5% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 0.78 | 0.08 | 761.72 | 7.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | accelerate | shared-session | not applicable | 1.92 | 0.09 | 1875.00 | 4.9% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 1.78 | 0.17 | 1738.28 | 19.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | accelerate | shared-session | not applicable | 2.57 | 0.33 | 2509.77 | 6.5% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 1.77 | 0.10 | 1728.52 | 5.3% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | accelerate | shared-session | not applicable | 0.80 | 0.04 | 781.25 | 4.8% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 0.69 | 0.05 | 673.83 | 6.5% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | accelerate | shared-session | not applicable | 1.26 | 0.03 | 1230.47 | 1.9% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 0.82 | 0.10 | 800.78 | 7.7% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | accelerate | shared-session | not applicable | 0.92 | 0.12 | 898.44 | 7.3% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 0.78 | 0.07 | 761.72 | 8.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | accelerate | shared-session | not applicable | 1.40 | 0.07 | 1367.19 | 19.0% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 0.94 | 0.07 | 917.97 | 6.4% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | accelerate | shared-session | not applicable | 2.19 | 0.05 | 2138.67 | 3.9% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 8.88 | 0.63 | 8671.88 | 4.8% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | accelerate | shared-session | not applicable | 2.92 | 0.11 | 2851.56 | 3.2% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 9.07 | 0.94 | 8857.42 | 6.9% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | accelerate | shared-session | not applicable | 0.84 | 0.05 | 820.31 | 5.0% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 8.43 | 0.60 | 8232.42 | 4.8% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | accelerate | shared-session | not applicable | 1.31 | 0.08 | 1279.30 | 19.8% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 7.46 | 0.39 | 7285.16 | 3.9% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | accelerate | shared-session | not applicable | 7.95 | 0.68 | 7763.67 | 5.1% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 18.03 | 0.60 | 17607.42 | 6.3% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | accelerate | shared-session | not applicable | 8.61 | 0.32 | 8408.20 | 3.9% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 17.97 | 0.69 | 17548.83 | 3.3% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | accelerate | shared-session | not applicable | 0.97 | 0.04 | 947.27 | 3.3% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 7.53 | 0.34 | 7353.52 | 3.1% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | accelerate | shared-session | not applicable | 1.39 | 0.05 | 1357.42 | 14.9% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 7.26 | 0.53 | 7089.84 | 4.6% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | accelerate | shared-session | not applicable | 1.10 | 0.07 | 1074.22 | 5.1% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 7.81 | 0.56 | 7626.95 | 4.7% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | accelerate | shared-session | not applicable | 1.64 | 0.21 | 1601.56 | 38.6% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 8.78 | 0.46 | 8574.22 | 3.8% | passed |

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
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | 128 | 0.71 | 0.03 | 0.734 | 0.738 | 3.8% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | 128 | 0.63 | 0.04 | 0.679 | 0.704 | 4.4% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | 128 | 2.77 | 0.16 | 3.013 | 3.243 | 6.2% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | 128 | 1.06 | 0.05 | 1.123 | 1.155 | 3.9% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | 128 | 0.74 | 0.03 | 0.749 | 0.790 | 3.3% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | 128 | 0.71 | 0.03 | 0.738 | 0.749 | 3.2% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | 128 | 2.95 | 0.21 | 3.170 | 3.323 | 4.8% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | 128 | 1.25 | 0.09 | 1.313 | 1.364 | 4.0% | passed |
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
  "latency_case_processes_seconds": 20.243465625000226,
  "separate_cache_diagnostics_seconds": 0.23992670899997393,
  "collection_total_seconds": 20.483392584000285
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 20.306867624999995,
  "separate_cache_diagnostics_seconds": 0.24316862500018033,
  "collection_total_seconds": 20.55003641700023
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 49.070968917,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 2.106248917,
  "validation_metadata_and_collection_seconds": 46.893837333,
  "command_before_report_seconds": 49.00008625
}
```

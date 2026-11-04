# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261104_223144`.

### run_t1

```yaml
timestamp: '2026-11-04T22:31:44Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 9192ee1056de2203697691c63cc9ea04b0c5298e
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu
    /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh 1 4 '
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
timestamp: '2026-11-04T22:32:26Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 9192ee1056de2203697691c63cc9ea04b0c5298e
  dirty: false
collection:
  command: 'BENCH_COMMAND_BUDGET_SECONDS=300 TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu
    /workspaces/tenferro-benchmark/scripts/run_cpu_session.sh 1 4 '
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
- Expected 114, selected 114, executed 48, unsupported 66, failed 0, skipped 0, missing 0, noisy 2.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 114 | 114 | 48 | 66 | 0 | 0 | 0 | 2 |

- Unsupported: `matmul_n2_count1024[faer]@t1`, `matmul_n4_count1024[faer]@t1`, `matmul_n8_count1024[faer]@t1`, `matmul_n16_count1024[faer]@t1`, `matmul_n32_count1024[faer]@t1`, `solve_n2_count1024[faer]@t1`, `solve_n4_count1024[faer]@t1`, `solve_n8_count1024[faer]@t1`, `solve_n16_count1024[faer]@t1`, `solve_n32_count1024[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t1`, `beinsum_f64_b1024_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m4n4k4_direct_into_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b16_m64n64k64_direct_into_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_alloc_auto[faer]@t1`, `bdot_f64_b64_m64n1k64_direct_into_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_alloc_auto[faer]@t1`, `bdot_f64_b1024_m4n4k4_canonical_into_auto[faer]@t1`, `hadamard_f64_m64n64_dot_alloc[faer]@t1`, `hadamard_f64_m64n64_einsum_alloc[faer]@t1`, `chain3_f64_n4_einsum_alloc[faer]@t1`, `stream_f64_fixed_len32_einsum_alloc[faer]@t1`, `stream_f64_mixed_len32_einsum_alloc[faer]@t1`, `stream_f64_fresh_len32_einsum_alloc[faer]@t1`, `stream_f64_strides_len32_einsum_alloc[faer]@t1`, `eager_stream_f64_fixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_mixed_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_fresh_len128_einsum3_alloc[faer]@t1`, `eager_stream_f64_strides_len128_einsum3_alloc[faer]@t1`, `matmul_n2_count1024[faer]@t4`, `matmul_n4_count1024[faer]@t4`, `matmul_n8_count1024[faer]@t4`, `matmul_n16_count1024[faer]@t4`, `matmul_n32_count1024[faer]@t4`, `solve_n2_count1024[faer]@t4`, `solve_n4_count1024[faer]@t4` (+26 more)
- Noisy: `solve_n16_count1024[blas]@t1`, `solve_n32_count1024[blas]@t1`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The worker count is recorded in each Rust row; tenferro-rs #2004 no longer exposes an execution mode, so a row that records one is historical. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

This independent-call loop lane uses concrete Tensor routes. Borrowed eager-session shape streams are reported separately below. Compiled trace still uses runtime-internal session entry and is not a borrowed-session route.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | blas | shared-session | not applicable | 2.36 | 0.06 | 2304.69 | 1.5% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 3.59 | 0.11 | 3505.86 | 4.3% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | blas | shared-session | not applicable | 3.55 | 0.08 | 3466.80 | 1.5% | passed |
| matmul_n16_count1024 | matmul | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 3.58 | 0.11 | 3496.09 | 3.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | blas | shared-session | not applicable | 1.47 | 0.01 | 1435.55 | 0.4% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 2.04 | 0.11 | 1992.19 | 6.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | blas | shared-session | not applicable | 2.68 | 0.01 | 2617.19 | 2.6% | passed |
| matmul_n2_count1024 | matmul | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 2.12 | 0.12 | 2070.31 | 5.0% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | blas | shared-session | not applicable | 6.73 | 0.13 | 6572.27 | 3.3% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 7.76 | 0.15 | 7578.12 | 1.9% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | blas | shared-session | not applicable | 12.28 | 0.17 | 11992.19 | 0.8% | passed |
| matmul_n32_count1024 | matmul | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 10.68 | 0.18 | 10429.69 | 2.0% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | blas | shared-session | not applicable | 1.52 | 0.01 | 1484.38 | 0.5% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.99 | 0.06 | 1943.36 | 8.2% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | blas | shared-session | not applicable | 2.62 | 0.01 | 2558.59 | 0.3% | passed |
| matmul_n4_count1024 | matmul | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.96 | 0.08 | 1914.06 | 5.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | blas | shared-session | not applicable | 1.67 | 0.01 | 1630.86 | 0.4% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 2.28 | 0.13 | 2226.56 | 6.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | blas | shared-session | not applicable | 2.69 | 0.02 | 2626.95 | 1.3% | passed |
| matmul_n8_count1024 | matmul | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 2.28 | 0.11 | 2226.56 | 6.2% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | blas | shared-session | not applicable | 4.34 | 1.11 | 4238.28 | 12.1% | passed |
| solve_n16_count1024 | solve | n16 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 21.90 | 0.49 | 21386.72 | 1.2% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | blas | shared-session | not applicable | 5.90 | 1.04 | 5761.72 | 9.5% | passed |
| solve_n16_count1024 | solve | n16 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 21.42 | 0.16 | 20917.97 | 0.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | blas | shared-session | not applicable | 1.24 | 0.01 | 1210.94 | 0.5% | passed |
| solve_n2_count1024 | solve | n2 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 16.92 | 0.18 | 16523.44 | 1.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | blas | shared-session | not applicable | 2.31 | 0.02 | 2255.86 | 0.7% | passed |
| solve_n2_count1024 | solve | n2 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 17.08 | 0.22 | 16679.69 | 1.0% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | blas | shared-session | not applicable | 19.20 | 4.33 | 18750.00 | 10.7% | passed |
| solve_n32_count1024 | solve | n32 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 38.04 | 0.45 | 37148.44 | 0.8% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | blas | shared-session | not applicable | 29.90 | 6.06 | 29199.22 | 9.9% | passed |
| solve_n32_count1024 | solve | n32 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 55.00 | 0.73 | 53710.94 | 1.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | blas | shared-session | not applicable | 1.37 | 0.02 | 1337.89 | 1.8% | passed |
| solve_n4_count1024 | solve | n4 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 17.17 | 0.23 | 16767.58 | 1.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | blas | shared-session | not applicable | 2.50 | 0.01 | 2441.41 | 0.3% | passed |
| solve_n4_count1024 | solve | n4 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 17.59 | 0.24 | 17177.73 | 1.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | blas | shared-session | not applicable | 2.48 | 0.02 | 2421.88 | 0.5% | passed |
| solve_n8_count1024 | solve | n8 | — | 1 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 18.12 | 0.27 | 17695.31 | 1.3% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | blas | shared-session | not applicable | 3.63 | 0.02 | 3544.92 | 0.6% | passed |
| solve_n8_count1024 | solve | n8 | — | 4 | faer | shared-session | not applicable | — | — | — | — | unsupported |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 18.56 | 0.38 | 18125.00 | 1.3% | passed |

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
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 1 | 128 | 1.24 | 0.02 | 1.267 | 1.325 | 2.0% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 1 | 128 | 1.26 | 0.02 | 1.272 | 1.288 | 1.0% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 1 | 128 | 6.33 | 0.25 | 6.545 | 6.990 | 3.7% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 1 | 128 | 1.98 | 0.02 | 1.998 | 2.010 | 0.8% | passed |
| eager_stream_f64_strides_len128_einsum3_alloc | faer | 1 | — | — | — | — | — | — | unsupported |
| stream_f64_fixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_mixed_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_fresh_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| stream_f64_strides_len32_einsum_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fixed_len128_einsum3_alloc | blas | 4 | 128 | 1.63 | 0.03 | 1.648 | 1.685 | 1.4% | passed |
| eager_stream_f64_fixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_mixed_len128_einsum3_alloc | blas | 4 | 128 | 1.62 | 0.02 | 1.645 | 1.725 | 1.8% | passed |
| eager_stream_f64_mixed_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_fresh_len128_einsum3_alloc | blas | 4 | 128 | 6.97 | 0.27 | 7.268 | 7.571 | 3.3% | passed |
| eager_stream_f64_fresh_len128_einsum3_alloc | faer | 4 | — | — | — | — | — | — | unsupported |
| eager_stream_f64_strides_len128_einsum3_alloc | blas | 4 | 128 | 2.42 | 0.02 | 2.439 | 2.466 | 0.7% | passed |
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
  "latency_case_processes_seconds": 37.49544349700045,
  "separate_cache_diagnostics_seconds": 0.5663010839998606,
  "collection_total_seconds": 38.06174494100014
}
```

Wall observations `wall_collection_t4.json`:
```json
{
  "latency_case_processes_seconds": 39.787678511000195,
  "separate_cache_diagnostics_seconds": 0.6153540280001835,
  "collection_total_seconds": 40.40303294000023
}
```

Wall observations `wall_command.json`:
```json
{
  "command_seconds_through_collection": 88.665502727,
  "scope": "command startup through collection; excludes report rendering and publication",
  "budget_seconds": 300.0,
  "budget_status": "within_budget"
}
```

Wall observations `wall_phases.json`:
```json
{
  "build_and_checkout_seconds": 2.917384308,
  "validation_metadata_and_collection_seconds": 85.683676177,
  "command_before_report_seconds": 88.601060485
}
```

# CPU shared-session matrix results

Raw data and provenance: `data/results/amd-cpu/cpu/session_matrix/20261006_183748`.

### run_t1

```yaml
timestamp: '2026-10-06T18:37:48Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
harness:
  path: /workspaces/t4a-2010-bench-migrate
  commit: unknown
  dirty: null
collection:
  command: 'TENFERRO_CPU_FEATURES=system-mkl BENCHMARK_TARGET_PROFILE=amd-cpu /workspaces/t4a-2010-bench-migrate/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 3
  selection_filter: null
  expected_cases: 29
  selected_cases: 29
```

### run_t4

```yaml
timestamp: '2026-10-06T18:38:28Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102
  dirty: false
  features:
  - system-mkl
  resolved_path: /workspaces/t4a-2010-bench-migrate/extern/tenferro-rs
harness:
  path: /workspaces/t4a-2010-bench-migrate
  commit: unknown
  dirty: null
collection:
  command: 'TENFERRO_CPU_FEATURES=system-mkl BENCHMARK_TARGET_PROFILE=amd-cpu /workspaces/t4a-2010-bench-migrate/scripts/run_cpu_session.sh
    1 4 '
  effort: standard
  warmups: 3
  runs: 15
  coverage: quick
  manifest_version: 3
  selection_filter: null
  expected_cases: 29
  selected_cases: 29
```

## Case status

- Manifest: `cpu/session_matrix` version 3, coverage `quick`, effort `standard`
- Expected 98, selected 98, executed 98, unsupported 0, failed 0, missing 0, noisy 4.
- Complete: **yes** — only executed cases count as covered; unsupported, failed, missing and unselected cases never do.
- Noisy: `matmul_n8_count1024[pytorch]@t4`, `matmul_n16_count1024[pytorch]@t4`, `beinsum_f64_b1024_m4n4k4_direct_alloc_auto[faer]@t4`, `chain3_f64_n4_einsum_alloc[faer]@t4`

## Loop of independent small calls

Each sample executes 1024 independent, distinct f64 matrix pairs; one wall-clock interval covers the whole loop. All inputs, output-container allocation, session entry, and initialization are outside timing. Outputs remain alive until timer stop. Every output is checked after timing (solve uses the residual).

tenferro enters exactly one backend session around all warmups and samples. Accelerate and faer use the same public operations and session scope. With tenferro-rs PR #1796 or later, managed BLAS and faer sessions both reuse the entered executor context. Earlier BLAS revisions included per-operation entry; consult the recorded tenferro commit. ProviderDefaultExclusive describes admission, not per-operation executor entry. The execution mode and worker count are recorded in each Rust row. 4-thread faer uses tenferro’s Rayon execution domain (inner kernel parallelism, not an outer parallel loop over matrices). PyTorch uses a Python loop over the same inputs, with its thread pools initialized before timing. Its Python dispatch cost is included. These are allocation-returning operations, not batched tensor APIs; the one-operation batched cases below are a different workload.

EagerTensor and compiled trace are not labeled shared-session: their current public interfaces do not accept this borrowed session and create internal sessions during execution. The old single-call measurements remain diagnostic evidence only.

| Case ID | Operation | Matrix | Operations | Threads | Provider | Route | Execution mode | Median total ms | IQR total ms | Median ns/op | CoV | Check |
|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 2.45 | 0.42 | 2392.58 | 8.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | faer | shared-session | Managed | 2.27 | 0.38 | 2216.80 | 8.2% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 2.48 | 0.28 | 2421.88 | 8.8% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 2.57 | 0.39 | 2509.77 | 7.5% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | faer | shared-session | Managed | 2.27 | 0.01 | 2216.80 | 1.7% | passed |
| matmul_n16_count1024 | matmul | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 2.75 | 0.43 | 2685.55 | 10.0% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.01 | 0.00 | 986.33 | 0.3% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | faer | shared-session | Managed | 0.86 | 0.00 | 839.84 | 0.4% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 1.41 | 0.03 | 1376.95 | 7.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.00 | 0.00 | 976.56 | 0.7% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | faer | shared-session | Managed | 0.85 | 0.00 | 830.08 | 1.2% | passed |
| matmul_n2_count1024 | matmul | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 1.50 | 0.27 | 1464.84 | 9.1% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 8.60 | 0.22 | 8398.44 | 3.8% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | faer | shared-session | Managed | 6.76 | 0.39 | 6601.56 | 3.4% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 5.87 | 0.30 | 5732.42 | 5.7% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 18.99 | 0.36 | 18544.92 | 1.2% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | faer | shared-session | Managed | 7.79 | 0.14 | 7607.42 | 1.6% | passed |
| matmul_n32_count1024 | matmul | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 14.11 | 1.69 | 13779.30 | 6.7% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.09 | 0.00 | 1064.45 | 0.2% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | faer | shared-session | Managed | 0.91 | 0.00 | 888.67 | 0.3% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 1.38 | 0.07 | 1347.66 | 9.8% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.09 | 0.18 | 1064.45 | 8.4% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | faer | shared-session | Managed | 0.92 | 0.01 | 898.44 | 2.2% | passed |
| matmul_n4_count1024 | matmul | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 1.38 | 0.03 | 1347.66 | 7.6% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.17 | 0.23 | 1142.58 | 9.3% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | faer | shared-session | Managed | 1.19 | 0.00 | 1162.11 | 0.2% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 1.47 | 0.27 | 1435.55 | 9.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.40 | 0.01 | 1367.19 | 0.4% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | faer | shared-session | Managed | 1.15 | 0.01 | 1123.05 | 0.9% | passed |
| matmul_n8_count1024 | matmul | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 1.48 | 0.29 | 1445.31 | 15.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 5.43 | 0.02 | 5302.73 | 6.2% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | faer | shared-session | Managed | 4.90 | 0.40 | 4785.16 | 4.7% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 1 | pytorch | shared-session | not applicable | 17.58 | 1.13 | 17167.97 | 6.6% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 4.56 | 0.08 | 4453.12 | 5.3% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | faer | shared-session | Managed | 4.83 | 0.00 | 4716.80 | 3.9% | passed |
| solve_n16_count1024 | solve | n16 | 1024 | 4 | pytorch | shared-session | not applicable | 21.16 | 0.68 | 20664.06 | 2.7% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.17 | 0.08 | 1142.58 | 7.1% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | faer | shared-session | Managed | 1.42 | 0.00 | 1386.72 | 0.4% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 1 | pytorch | shared-session | not applicable | 13.87 | 0.47 | 13544.92 | 3.2% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.22 | 0.01 | 1191.41 | 0.6% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | faer | shared-session | Managed | 1.43 | 0.00 | 1396.48 | 0.2% | passed |
| solve_n2_count1024 | solve | n2 | 1024 | 4 | pytorch | shared-session | not applicable | 13.80 | 0.18 | 13476.56 | 1.4% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 21.76 | 1.99 | 21250.00 | 6.0% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | faer | shared-session | Managed | 17.74 | 1.66 | 17324.22 | 6.5% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 1 | pytorch | shared-session | not applicable | 35.91 | 3.91 | 35068.36 | 7.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 29.62 | 0.34 | 28925.78 | 1.1% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | faer | shared-session | Managed | 17.83 | 0.14 | 17412.11 | 0.6% | passed |
| solve_n32_count1024 | solve | n32 | 1024 | 4 | pytorch | shared-session | not applicable | 42.83 | 4.03 | 41826.17 | 5.4% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.07 | 0.17 | 1044.92 | 8.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | faer | shared-session | Managed | 1.61 | 0.01 | 1572.27 | 4.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 1 | pytorch | shared-session | not applicable | 14.05 | 0.96 | 13720.70 | 5.0% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 1.27 | 0.01 | 1240.23 | 0.5% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | faer | shared-session | Managed | 1.67 | 0.00 | 1630.86 | 0.2% | passed |
| solve_n4_count1024 | solve | n4 | 1024 | 4 | pytorch | shared-session | not applicable | 14.97 | 2.74 | 14619.14 | 9.0% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | blas | shared-session | ProviderDefaultExclusive | 1.91 | 0.26 | 1865.23 | 7.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | faer | shared-session | Managed | 2.35 | 0.26 | 2294.92 | 7.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 1 | pytorch | shared-session | not applicable | 14.48 | 0.53 | 14140.62 | 5.5% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | blas | shared-session | ProviderDefaultExclusive | 2.08 | 0.28 | 2031.25 | 6.9% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | faer | shared-session | Managed | 2.35 | 0.38 | 2294.92 | 8.3% | passed |
| solve_n8_count1024 | solve | n8 | 1024 | 4 | pytorch | shared-session | not applicable | 14.74 | 0.23 | 14394.53 | 5.1% | passed |

## One-operation batched routes (tenferro-rs #1946 B2)

One public call per operation; many operations per wall-clock interval (calibrated toward the suite `min_runtime_ms`, bounded to 64 MiB of retained outputs); one entered backend session around warmup, calibration and all samples; policy overrides applied before session entry (backend) or around all samples (scoped). Allocating routes include their intrinsic output allocation; `_into` routes reuse one destination allocated outside timing. Inputs, views, the first validated call and all numerical checks are outside timing. Route columns come from the case manifest; provider-call/lane mechanics are in the separate `cpu/route_contract` diagnostics, never in these timings. NOISY (CoV > 10%) is descriptive only.

| Case ID | Route (path / output / repr / layout) | Policy | Dtype | Batch×m×n×k | Threads | Workers | Ops/sample | Median ns/op | IQR ns/op | CoV | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 221956.80 | 930.00 | 0.4% | passed |
| bdot_f64_b1024_m4n4k4_canonical_alloc_auto | concrete / allocating / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 7 | 73582.29 | 1220.00 | 1.4% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 1 | 1 | 5 | 181429.60 | 284.00 | 0.2% | passed |
| bdot_f64_b1024_m4n4k4_canonical_into_auto | concrete / reused / view / canonical-copy | auto | f64 | 1024×4×4×4 | 4 | 4 | 8 | 55966.75 | 1130.69 | 2.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 12 | 130311.50 | 504.21 | 0.3% | passed |
| bdot_f64_b1024_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 12 | 51089.83 | 1994.62 | 2.4% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 8 | 71942.25 | 545.12 | 0.7% | passed |
| bdot_f64_b1024_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 12 | 26795.83 | 383.79 | 3.7% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 181305.75 | 642.50 | 2.4% | passed |
| bdot_f64_b16_m64n64k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 130984.00 | 5151.83 | 2.4% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 1 | 1 | 4 | 217589.25 | 522.38 | 0.2% | passed |
| bdot_f64_b16_m64n64k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 16×64×64×64 | 4 | 4 | 3 | 128050.33 | 3773.50 | 1.7% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 87 | 9422.60 | 20.05 | 1.3% | passed |
| bdot_f64_b64_m4n4k4_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 109 | 9614.51 | 22.29 | 2.4% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 1 | 1 | 72 | 5719.19 | 49.10 | 0.6% | passed |
| bdot_f64_b64_m4n4k4_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×4×4×4 | 4 | 4 | 151 | 5680.83 | 25.07 | 0.3% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 14 | 51650.14 | 1641.14 | 2.2% | passed |
| bdot_f64_b64_m64n1k64_direct_alloc_auto | concrete / allocating / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 18 | 35887.22 | 1424.25 | 3.1% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 1 | 1 | 5 | 50175.60 | 1150.00 | 4.9% | passed |
| bdot_f64_b64_m64n1k64_direct_into_auto | concrete / reused / owned / direct | auto | f64 | 64×64×1×64 | 4 | 4 | 16 | 36444.81 | 1303.19 | 2.5% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 10 | 139211.30 | 1223.00 | 1.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_alloc_auto | concrete-einsum / allocating / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 11 | 69834.91 | 12707.64 | 11.9% | passed (NOISY) |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 1 | 1 | 13 | 72593.77 | 496.92 | 0.7% | passed |
| beinsum_f64_b1024_m4n4k4_direct_into_auto | concrete-einsum / reused / owned / direct | auto | f64 | 1024×4×4×4 | 4 | 4 | 12 | 27918.33 | 554.62 | 4.3% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 1 | 1 | 12 | 23633.25 | 694.63 | 2.0% | passed |
| chain3_f64_n4_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×4×4×4 | 4 | 4 | 20 | 23414.70 | 5750.65 | 20.8% | passed (NOISY) |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 70 | 18896.57 | 83.36 | 1.7% | passed |
| hadamard_f64_m64n64_dot_alloc | concrete / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 133 | 20029.19 | 385.09 | 5.3% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 1 | 1 | 53 | 25552.28 | 196.80 | 0.8% | passed |
| hadamard_f64_m64n64_einsum_alloc | concrete-einsum / allocating / owned / direct | auto | f64 | 1×64×64×1 | 4 | 4 | 55 | 25392.78 | 62.64 | 0.2% | passed |
| stream_f64_fixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9869.38 | 123.73 | 1.1% | passed |
| stream_f64_fixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 10054.38 | 328.94 | 2.5% | passed |
| stream_f64_fresh_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 10329.06 | 104.20 | 1.0% | passed |
| stream_f64_fresh_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9703.72 | 311.58 | 2.8% | passed |
| stream_f64_mixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 1 | 1 | 32 | 9667.16 | 216.56 | 1.6% | passed |
| stream_f64_mixed_len32_einsum_alloc | concrete-einsum / allocating / view / direct | auto | f64 | None×None×None×None | 4 | 4 | 32 | 9965.00 | 220.16 | 1.5% | passed |
| stream_f64_strides_len32_einsum_alloc | concrete-einsum / allocating / view / direct-strided | auto | f64 | None×None×None×None | 1 | 1 | 32 | 10330.00 | 281.12 | 1.7% | passed |
| stream_f64_strides_len32_einsum_alloc | concrete-einsum / allocating / view / direct-strided | auto | f64 | None×None×None×None | 4 | 4 | 32 | 10420.00 | 163.14 | 1.6% | passed |

# M5 Max CPU refresh after timing corrections

All seven CPU reports were regenerated on 2026-09-13 on native macOS, using
Apple M5 Max and timing policy revision 2. Collection ran sequentially at
1 and 4 threads. GPU and other CPU hosts were not rerun.

- Benchmark source: `15c25b1c003c993e0a5b4f4dc05a97a4a836a282`.
- tenferro-rs: `a793c2e95693f053722fbff8d0db25722336c21f`, clean `main`,
  updated with `git pull --ff-only` before collection.
- tenferro and PyTorch use Accelerate; JAX uses XLA CPU; Julia uses OpenBLAS.
- PyTorch 2.12.0, JAX/jaxlib 0.10.1, Julia 1.12.5, OMEinsum.jl 0.9.4,
  Strided.jl 2.6.1. Full provider and thread environments are in each run YAML.
- Input/session/plan preparation and returned-output destruction are outside
  timing. Setup diagnostics were disabled; the idle-host guard was not bypassed.

Command, run in this repository after updating the external tenferro checkout:

```sh
BENCHMARK_TARGET_PROFILE=mac-cpu PUBLICATION_GATE_PROFILE=full ./scripts/run_all.sh 1 4
```

The main suites and public API use 3 warmups and 15 measured runs. Permutation
uses the runner defaults: Rust 3/15, Julia 5/40. Small-work uses calibrated
iteration batches with 15 samples per case. See the raw records for each
runner's sampling details.

## Reports and raw runs

The main runner's latest einsum, CPU ops and linalg AD reports show the last
thread count (4). Both thread counts' generated reports and raw data are
preserved below. FFT, public API, small-work and permutation reports include
both thread counts.

| Suite | 1 thread | 4 threads |
|---|---|---|
| Einsum | [Report](../data/results/mac-cpu/cpu/einsum/20260913_141327/report.md) | [Report](../data/results/mac-cpu/cpu/einsum/20260913_142108/report.md) |
| CPU ops | [Report](../data/results/mac-cpu/cpu/einsum/20260913_141327/cpu_ops_report.md) | [Report](../data/results/mac-cpu/cpu/einsum/20260913_142108/cpu_ops_report.md) |
| Linalg JVP/VJP | [Report](../data/results/mac-cpu/cpu/einsum/20260913_141327/linalg_jvp_vjp_report.md) | [Report](../data/results/mac-cpu/cpu/einsum/20260913_142108/linalg_jvp_vjp_report.md) |
| FFT | [Combined report](../result/mac-cpu/cpu/fft.md) | [Raw run](../data/results/mac-cpu/cpu/fft/20260913_142656/) |
| Public API | [Combined report](../result/mac-cpu/cpu/public_api.md) | [Raw run](../data/results/mac-cpu/cpu/public_api/20260913_142731/) |
| Small-work | [Combined report](../result/mac-cpu/cpu/small_work.md) | [Raw run](../data/results/mac-cpu/cpu/small_work/20260913_053129/) |
| Permutation | [Combined report](../result/mac-cpu/cpu/permutation.md) | [Raw run](../data/results/mac-cpu/cpu/permutation/20260913_143137/) |

Directory timestamps use the existing runners' conventions: small-work uses
UTC, while the other directory names use local time (JST). The YAML timestamps
identify the actual collection time unambiguously.

## Validation

The collection command exited successfully. Run metadata and permutation JSONL
passed repository schema validation. All run metadata records revision 2 and
the same clean tenferro commit. Case/status coverage matches across thread
counts, with no duplicate case/backend keys in the collected CSV/JSONL files.

Per thread count:

- Einsum: 19 instances × 2 path strategies, all five backend columns populated.
- CPU ops: 1,016 successful rows, including the linalg JVP/VJP report's rows.
- FFT: 32 successful rows; 16 one-shot planning rows skipped under the policy.
- Public API: 429 successful rows; 18 unsupported rows with recorded reasons.
  These include unavailable trace surfaces and full-matrices SVD on Accelerate.
- Small-work: 55 ordinary-operation cases, all correctness checks passed.
- Permutation: 38 successful rows with correctness checks passed; 8 HPTT rows
  skipped (the optional feature was disabled, and its one-shot API would not
  satisfy the prepared-plan timing rule).

Earlier runs on this date used different timing scopes. Their timings remain
preserved for provenance and must not be treated as revision 2 regression baselines.
